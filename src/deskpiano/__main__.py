import mido
import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import argparse

# Add FastAPI app
app = FastAPI()

# === Constants ===
SAMPLE_RATE = 44100
BLOCKSIZE = 64  # smaller = lower latency

# Reverb parameters
REVERB_DELAY = int(0.13 * SAMPLE_RATE)  # 30ms delay
NUM_REVERBS = 4  # Number of delay lines
REVERB_BUFFER_SIZE = REVERB_DELAY * NUM_REVERBS

# === Instrument Configurations ===
PIANO_CONFIG = {
    'attack_time': 0.1,
    'decay_time': 0.1,
    'sustain_level': 0.7,
    'release_time': 0.2,
    'harmonics': [(1.0, 1.0)],  # Just fundamental frequency
}

HARPSICHORD_CONFIG = {
    'attack_time': 0.002,
    'decay_time': 0.1,
    'sustain_level': 0.1,
    'release_time': 0.5,
    'harmonics': [
        (1.0, 1.0),    # fundamental
        (0.8, 2.0),    # octave
        (0.5, 3.0),    # twelfth
        (0.3, 4.0),    # double octave
        (0.2, 5.0),    # major third + 2 octaves
        (0.1, 6.0),    # fifth + 2 octaves
    ],
}

# Define request/response models with Pydantic
class SynthParams(BaseModel):
    use_harpsichord: bool = False
    use_reverb: bool = True
    volume: float = 0.1
    output_gain: float = 5.0
    reverb_decay: float = 0.5
    filter_cutoff: float = 2000
    high_pass_cutoff: float = 20

# Global parameters instance
params = SynthParams()

# Set active configuration
ACTIVE_CONFIG = HARPSICHORD_CONFIG if params.use_harpsichord else PIANO_CONFIG

# Use these variables in place of the original constants
ATTACK_TIME = ACTIVE_CONFIG['attack_time']
DECAY_TIME = ACTIVE_CONFIG['decay_time']
SUSTAIN_LEVEL = ACTIVE_CONFIG['sustain_level']
RELEASE_TIME = ACTIVE_CONFIG['release_time']
HARMONICS = ACTIVE_CONFIG['harmonics']

# === State ===
active_notes = {}  # {midi_note: (start_time, velocity)}
note_timestamps = {}  # {midi_note: midi_receive_time}
last_output_low = 0.0  # For low-pass filter
last_output_high = 0.0  # For high-pass filter
last_input_high = 0.0  # For high-pass filter
lock = threading.Lock()

reverb_buffers = [deque([0.0] * REVERB_DELAY, maxlen=REVERB_DELAY) for _ in range(NUM_REVERBS)]
reverb_gains = [params.reverb_decay ** (i + 1) for i in range(NUM_REVERBS)]

# Add API endpoints
@app.get("/params")
def get_params():
    return params

@app.post("/params")
def update_params(new_params: SynthParams):
    global ACTIVE_CONFIG, ATTACK_TIME, DECAY_TIME, SUSTAIN_LEVEL, RELEASE_TIME, HARMONICS, params
    
    # Update only provided values
    update_data = new_params.dict(exclude_unset=True)
    current_data = params.dict()
    updated_data = {**current_data, **update_data}
    params = SynthParams(**updated_data)
    
    if 'use_harpsichord' in update_data:
        ACTIVE_CONFIG = HARPSICHORD_CONFIG if params.use_harpsichord else PIANO_CONFIG
        ATTACK_TIME = ACTIVE_CONFIG['attack_time']
        DECAY_TIME = ACTIVE_CONFIG['decay_time']
        SUSTAIN_LEVEL = ACTIVE_CONFIG['sustain_level']
        RELEASE_TIME = ACTIVE_CONFIG['release_time']
        HARMONICS = ACTIVE_CONFIG['harmonics']
    
    if 'reverb_decay' in update_data:
        global reverb_gains
        reverb_gains = [params.reverb_decay ** (i + 1) for i in range(NUM_REVERBS)]
    
    if 'filter_cutoff' in update_data or 'high_pass_cutoff' in update_data:
        set_filter_cutoffs(params.filter_cutoff, params.high_pass_cutoff)
    
    return {"status": "success"}

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
            env = envelope(current_time, start_time, ATTACK_TIME, DECAY_TIME, 
                         SUSTAIN_LEVEL, RELEASE_TIME, release_time)
            
            if env <= 0:
                notes_to_remove.append(note)
                continue
            
            # Sum all harmonics for this note
            wave = np.zeros_like(t)
            for amplitude, harmonic in HARMONICS:
                phase = 2 * np.pi * (freq * harmonic) * t
                if params.use_harpsichord and harmonic > 1:
                    phase += 0.0001 * harmonic
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

    # Apply distortion for harpsichord
    if params.use_harpsichord:
        filtered = np.tanh(filtered * 1.1)

    # Apply reverb
    if params.use_reverb:
        filtered = apply_reverb(filtered)

    # Final output gain and clipping
    filtered = np.clip(filtered * params.output_gain, -1.0, 1.0)
    
    outdata[:] = filtered.reshape(-1, 1)
    audio_callback.frame += frames

def set_filter_cutoffs(low_freq, high_freq):
    global FILTER_ALPHA, HIGH_PASS_ALPHA
    FILTER_ALPHA = np.exp(-2 * np.pi * low_freq / SAMPLE_RATE)
    HIGH_PASS_ALPHA = np.exp(-2 * np.pi * high_freq / SAMPLE_RATE)

async def midi_loop():
    input_names = mido.get_input_names()
    if not input_names:
        print("No MIDI input devices found.")
        return

    print(f"Using MIDI input: {input_names[1]}")
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

async def run_server():
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def run_audio_stream():
    # We still need to run the audio stream in a separate thread
    # because sounddevice's callback needs to be real-time
    stream = sd.OutputStream(
        channels=1,
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype='float32'
    )
    
    with stream:
        print("Audio stream started. Press Ctrl+C to quit.")
        while True:
            await asyncio.sleep(1)

async def main_loop():
    parser = argparse.ArgumentParser(description='DeskPiano - A software synthesizer')
    parser.add_argument('--no-server', action='store_true', 
                       help='Run without the FastAPI server')
    args = parser.parse_args()

    audio_callback.frame = 0
    set_filter_cutoffs(2000, 20)

    try:
        # Create tasks for MIDI handling and audio stream
        tasks = [
            asyncio.create_task(midi_loop()),
            asyncio.create_task(run_audio_stream())
        ]

        # Add server task if --no-server is not specified
        if not args.no_server:
            tasks.append(asyncio.create_task(run_server()))
            print("API server running on http://localhost:5000")

        # Wait for all tasks to complete (or KeyboardInterrupt)
        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        print("\nExiting...")
        # Cancel all running tasks
        for task in tasks:
            task.cancel()
        # Wait for tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)

def main():
    asyncio.run(main_loop())
