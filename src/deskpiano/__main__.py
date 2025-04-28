import mido
import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time
import asyncio
import argparse
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
SAMPLE_RATE = 44100
BLOCKSIZE = 64  # smaller = lower latency

# Reverb parameters
REVERB_DELAY = int(0.13 * SAMPLE_RATE)  # 30ms delay
NUM_REVERBS = 4  # Number of delay lines
REVERB_BUFFER_SIZE = REVERB_DELAY * NUM_REVERBS

# Global parameters
class SynthParams:
    def __init__(self):
        self.use_reverb = True
        self.volume = 0.1
        self.output_gain = 5.0
        self.reverb_decay = 0.5
        self.filter_cutoff = 2000
        self.high_pass_cutoff = 20
        self.active_config = None

params = SynthParams()

# === State ===
active_notes = {}  # {midi_note: (start_time, velocity)}
note_timestamps = {}  # {midi_note: midi_receive_time}
last_output_low = 0.0  # For low-pass filter
last_output_high = 0.0  # For high-pass filter
last_input_high = 0.0  # For high-pass filter
lock = threading.Lock()

reverb_buffers = [deque([0.0] * REVERB_DELAY, maxlen=REVERB_DELAY) for _ in range(NUM_REVERBS)]
reverb_gains = [params.reverb_decay ** (i + 1) for i in range(NUM_REVERBS)]

def load_instrument_config(json_path):
    logger.info(f"Loading instrument config from {json_path}")
    with open(json_path, 'r') as f:
        config = json.load(f)
    return {
        'attack_time': config['attack_time'],
        'decay_time': config['decay_time'],
        'sustain_level': config['sustain_level'],
        'release_time': config['release_time'],
        'harmonics': config['harmonics'],
    }

def set_filter_cutoffs(cutoff_low, cutoff_high):
    global FILTER_ALPHA, HIGH_PASS_ALPHA
    FILTER_ALPHA = np.exp(-2 * np.pi * cutoff_low / SAMPLE_RATE)
    HIGH_PASS_ALPHA = np.exp(-2 * np.pi * cutoff_high / SAMPLE_RATE)

def apply_reverb(dry_signal):
    wet_signal = np.zeros_like(dry_signal)
    
    for i, sample in enumerate(dry_signal):
        # Add the dry sample to each reverb buffer
        for buf, gain in zip(reverb_buffers, reverb_gains):
            # Get the delayed sample and add it to the output
            delayed = buf[0]
            wet_signal[i] += delayed * gain
            
            # Update the buffer with the new sample
            buf.append(sample + delayed * 0.5)
    
    # Mix dry and wet signals
    return dry_signal * 0.7 + wet_signal * 0.3

def frequency_from_midi_note(note):
    return 440.0 * (2 ** ((note - 69) / 12))

def envelope(time, start_time, attack_time, decay_time, sustain_level, release_time, release_start=None):
    elapsed = time - start_time
    
    # If note is released, calculate release envelope
    if release_start is not None:
        release_elapsed = time - release_start
        if release_elapsed >= release_time:
            return 0
        # Linear fade-out for release phase
        return sustain_level * (1 - release_elapsed / release_time)
    
    # Attack phase
    if elapsed < attack_time:
        return elapsed / attack_time
    # Decay phase
    elif elapsed < attack_time + decay_time:
        return 1 - (1 - sustain_level) * (elapsed - attack_time) / decay_time
    # Sustain phase
    else:
        return sustain_level

def audio_callback(outdata, frames, time_info, status):
    global last_output_low, last_output_high, last_input_high
    t = (np.arange(frames) + audio_callback.frame) / SAMPLE_RATE
    out = np.zeros(frames, dtype=np.float32)
    current_time = time.time()
    
    with lock:
        notes_to_remove = []
        
        for note, (start_time, velocity, release_time) in active_notes.items():
            freq = frequency_from_midi_note(note)
            env = envelope(current_time, start_time, 
                         params.active_config['attack_time'],
                         params.active_config['decay_time'],
                         params.active_config['sustain_level'],
                         params.active_config['release_time'],
                         release_time)
            
            if env <= 0:
                notes_to_remove.append(note)
                continue
            
            # Sum all harmonics for this note
            wave = np.zeros_like(t)
            for amplitude, harmonic in params.active_config['harmonics']:
                phase = 2 * np.pi * (freq * harmonic) * t
                wave += amplitude * np.sin(phase)
            
            # Apply envelope and velocity
            wave = wave * env * velocity * params.volume
            out += wave
            
        for note in notes_to_remove:
            del active_notes[note]

    # Apply filters
    filtered = np.zeros_like(out)
    for i in range(len(out)):
        new_low = out[i] * (1 - FILTER_ALPHA) + last_output_low * FILTER_ALPHA
        if not np.isnan(new_low) and np.abs(new_low) < 100:
            last_output_low = new_low
        
        high_pass = (last_output_high * HIGH_PASS_ALPHA + 
                    last_output_low - last_input_high)
        if not np.isnan(high_pass) and np.abs(high_pass) < 100:
            last_output_high = high_pass
            last_input_high = last_output_low
        
        filtered[i] = high_pass

    # Apply reverb if enabled
    if params.use_reverb:
        filtered = apply_reverb(filtered)

    # Final output gain and clipping
    outdata[:] = np.clip(filtered * params.output_gain, -1, 1).reshape(-1, 1)
    audio_callback.frame += frames

audio_callback.frame = 0

async def midi_loop():
    input_names = mido.get_input_names()
    if not input_names:
        logger.error("No MIDI input devices found.")
        return

    logger.info(f"Using MIDI input: {input_names[1]}")
    with mido.open_input(input_names[1]) as port:
        while True:
            # Non-blocking check for MIDI messages
            for msg in port.iter_pending():
                now = time.time()
                if msg.type == 'note_on' and msg.velocity > 0:
                    with lock:
                        velocity = msg.velocity / 127.0 * params.volume
                        active_notes[msg.note] = (now, velocity, None)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    with lock:
                        if msg.note in active_notes:
                            start_time, velocity, _ = active_notes[msg.note]
                            active_notes[msg.note] = (start_time, velocity, now)
            
            await asyncio.sleep(0.001)  # Small sleep to prevent CPU hogging

async def run_audio_stream():
    stream = sd.OutputStream(
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype='float32'
    )
    
    with stream:
        logger.info("Audio stream started. Press Ctrl+C to quit.")
        while True:
            await asyncio.sleep(1)

async def main_loop():
    parser = argparse.ArgumentParser(description='DeskPiano - A software synthesizer')
    parser.add_argument('--instrument', type=str, default='piano.json', help='JSON file containing instrument configuration')
    parser.add_argument('--no-reverb', action='store_false', dest='reverb', help='Disable reverb')
    parser.add_argument('--volume', type=float, default=0.1, help='Volume (0.0-1.0)')
    args = parser.parse_args()

    params.use_reverb = args.reverb
    params.volume = args.volume

    logger.info(f"Loading instrument from: {args.instrument}")
    # Load instrument configuration
    instrument_path = Path(args.instrument)
    if not instrument_path.is_absolute():
        # First check in the current directory
        if instrument_path.exists():
            pass
        else:
            # Then check in the package's instruments directory
            package_dir = Path(__file__).parent
            default_path = package_dir / 'instruments' / args.instrument
            if default_path.exists():
                instrument_path = default_path
                logger.info(f"Using instrument from package path: {default_path}")
            else:
                logger.error(f"Could not find instrument file: {args.instrument}")
                logger.error(f"Searched in:\n- {instrument_path}\n- {default_path}")
                return

    set_filter_cutoffs(params.filter_cutoff, params.high_pass_cutoff)

    params.active_config = load_instrument_config(instrument_path)
    if params.active_config is None:
        logger.error("Failed to load instrument configuration")
        return

    try:
        tasks = [
            asyncio.create_task(midi_loop()),
            asyncio.create_task(run_audio_stream())
        ]
        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        logger.info("\nExiting...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

def main():
    logger.info("Starting DeskPiano...")
    try:
        asyncio.run(main_loop())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()

