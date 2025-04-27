import mido
import sounddevice as sd
import numpy as np
import threading
import time

# === Constants ===
SAMPLE_RATE = 44100
BLOCKSIZE = 64  # smaller = lower latency
VOLUME = 0.15

FILTER_CUTOFF = 2000  # Cutoff frequency in Hz
FILTER_ALPHA = np.exp(-2 * np.pi * FILTER_CUTOFF / SAMPLE_RATE)
HIGH_PASS_CUTOFF = 20  # Cutoff frequency in Hz
HIGH_PASS_ALPHA = np.exp(-2 * np.pi * HIGH_PASS_CUTOFF / SAMPLE_RATE)

# === ADSR Envelope Parameters ===
DECAY_TIME = 0.2  # in seconds
ATTACK_TIME = 0.1  # in seconds
SUSTAIN_LEVEL = 0.3  # 0 to 1
RELEASE_TIME = 0.2  # in seconds

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
        max_amplitude = 0.0
        
        # First pass: calculate maximum amplitude
        for note, (start_time, velocity, release_time) in active_notes.items():
            env = envelope(current_time, start_time, ATTACK_TIME, DECAY_TIME, 
                         SUSTAIN_LEVEL, RELEASE_TIME, release_time)
            if env <= 0:
                notes_to_remove.append(note)
                continue
            max_amplitude = max(max_amplitude, velocity * env)
        
        # Second pass: generate audio with normalized amplitudes
        for note, (start_time, velocity, release_time) in active_notes.items():
            freq = frequency_from_midi_note(note)
            phase = 2 * np.pi * freq * t
            
            env = envelope(current_time, start_time, ATTACK_TIME, DECAY_TIME, 
                         SUSTAIN_LEVEL, RELEASE_TIME, release_time)
            
            if env <= 0:
                continue
                
            adjusted_velocity = (velocity * env) / max(max_amplitude, 1.0)
            wave = np.sin(phase) * adjusted_velocity * VOLUME
            out += wave
            
        # Clean up finished notes
        for note in notes_to_remove:
            del active_notes[note]

    # Apply filters in series (low-pass then high-pass)
    filtered = np.zeros_like(out)
    for i in range(len(out)):
        # Low-pass filter
        last_output_low = out[i] * (1 - FILTER_ALPHA) + last_output_low * FILTER_ALPHA
        
        # High-pass filter
        high_pass = (last_output_high * HIGH_PASS_ALPHA + 
                    last_output_low - last_input_high)
        last_input_high = last_output_low
        last_output_high = high_pass
        
        filtered[i] = high_pass

    out = np.clip(filtered, -1.0, 1.0)
    outdata[:] = out.reshape(-1, 1)
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

