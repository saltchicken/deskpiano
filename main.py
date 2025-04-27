import mido
import sounddevice as sd
import numpy as np
import threading
import time

# === Constants ===
SAMPLE_RATE = 44100
BLOCKSIZE = 64  # smaller = lower latency
VOLUME = 0.2

FILTER_CUTOFF = 2000  # Cutoff frequency in Hz
FILTER_ALPHA = np.exp(-2 * np.pi * FILTER_CUTOFF / SAMPLE_RATE)
HIGH_PASS_CUTOFF = 20  # Cutoff frequency in Hz
HIGH_PASS_ALPHA = np.exp(-2 * np.pi * HIGH_PASS_CUTOFF / SAMPLE_RATE)

# === ADSR Envelope Parameters ===
USE_HARPSICHORD = True  # Set to True for harpsichord, False for original piano

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
    'release_time': 0.05,
    'harmonics': [
        (1.0, 1.0),    # fundamental
        (0.8, 2.0),    # octave
        (0.5, 3.0),    # twelfth
        (0.3, 4.0),    # double octave
        (0.2, 5.0),    # major third + 2 octaves
        (0.1, 6.0),    # fifth + 2 octaves
    ],
}

# Set active configuration
ACTIVE_CONFIG = HARPSICHORD_CONFIG if USE_HARPSICHORD else PIANO_CONFIG

# Use these variables in place of the original constants
ATTACK_TIME = ACTIVE_CONFIG['attack_time']
DECAY_TIME = ACTIVE_CONFIG['decay_time']
SUSTAIN_LEVEL = ACTIVE_CONFIG['sustain_level']
RELEASE_TIME = ACTIVE_CONFIG['release_time']
HARMONICS = ACTIVE_CONFIG['harmonics']

# === State ===
active_notes = {}  # {midi_note: (start_time, velocity)}
note_timestamps = {}  # {midi_note: midi_receive_time}
latency_measurements = []  # list of latencies (seconds)
last_output_low = 0.0  # For low-pass filter
last_output_high = 0.0  # For high-pass filter
last_input_high = 0.0  # For high-pass filter
lock = threading.Lock()

# === Audio Setup ===
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
                # Add slight detuning only for harpsichord
                if USE_HARPSICHORD and harmonic > 1:
                    phase += 0.0001 * harmonic
                wave += amplitude * np.sin(phase)
            
            # Apply envelope and velocity
            wave = wave * env * velocity * VOLUME
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

    # Apply distortion only for harpsichord
    if USE_HARPSICHORD:
        filtered = np.tanh(filtered * 1.1)

    outdata[:] = filtered.reshape(-1, 1)
    audio_callback.frame += frames

# Add this function to dynamically control both filters
def set_filter_cutoffs(low_freq, high_freq):
    global FILTER_ALPHA, HIGH_PASS_ALPHA
    FILTER_ALPHA = np.exp(-2 * np.pi * low_freq / SAMPLE_RATE)
    HIGH_PASS_ALPHA = np.exp(-2 * np.pi * high_freq / SAMPLE_RATE)

def midi_thread():
    input_names = mido.get_input_names()
    if not input_names:
        print("No MIDI input devices found.")
        return

    print(f"Using MIDI input: {input_names[1]}")
    with mido.open_input(input_names[1]) as port:
        for msg in port:
            now = time.time()
            if msg.type == 'note_on' and msg.velocity > 0:
                with lock:
                    velocity = msg.velocity / 127.0 * VOLUME
                    active_notes[msg.note] = (now, velocity, None)  # None means note is not released
                    note_timestamps[msg.note] = now
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                with lock:
                    if msg.note in active_notes:
                        start_time, velocity, _ = active_notes[msg.note]
                        active_notes[msg.note] = (start_time, velocity, now)  # Store release time

# === Main ===
if __name__ == "__main__":
    audio_callback.frame = 0
    midi_thread_obj = threading.Thread(target=midi_thread, daemon=True)
    midi_thread_obj.start()

    set_filter_cutoffs(2000, 20)

    try:
        with sd.OutputStream(
            channels=1,
            callback=audio_callback,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            dtype='float32'
        ):
            print("Audio stream started. Press Ctrl+C to quit.")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting...")
        if latency_measurements:
            avg_latency = sum(latency_measurements) / len(latency_measurements)
            min_latency = min(latency_measurements)
            max_latency = max(latency_measurements)
            print(f"\nLatency Results:")
            print(f"- Average: {avg_latency * 1000:.2f} ms")
            print(f"- Minimum: {min_latency * 1000:.2f} ms")
            print(f"- Maximum: {max_latency * 1000:.2f} ms")
        else:
            print("No latency measurements recorded.")

