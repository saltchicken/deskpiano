import mido
import sounddevice as sd
import numpy as np
import threading
import time

# === Constants ===
SAMPLE_RATE = 48000
BLOCKSIZE = 1024  # smaller = lower latency
VOLUME = 0.01

# === ADSR Envelope Parameters ===
ATTACK_TIME = 0.1  # in seconds
DECAY_TIME = 0.2  # in seconds
SUSTAIN_LEVEL = 0.7  # 0 to 1
RELEASE_TIME = 0.2  # in seconds

# === State ===
active_notes = {}  # {midi_note: (start_time, velocity)}
note_timestamps = {}  # {midi_note: midi_receive_time}
latency_measurements = []  # list of latencies (seconds)
lock = threading.Lock()

# === Audio Setup ===
def frequency_from_midi_note(note):
    return 440.0 * (2 ** ((note - 69) / 12))

def envelope(time, start_time, attack_time, decay_time, sustain_level, release_time):
    elapsed = time - start_time

    # Attack phase
    if elapsed < attack_time:
        return elapsed / attack_time
    # Decay phase
    elif elapsed < attack_time + decay_time:
        return 1 - (1 - sustain_level) * (elapsed - attack_time) / decay_time
    # Sustain phase
    elif elapsed < attack_time + decay_time + sustain_level:
        return sustain_level
    # Release phase
    elif elapsed < attack_time + decay_time + sustain_level + release_time:
        return sustain_level * (1 - (elapsed - attack_time - decay_time - sustain_level) / release_time)
    else:
        return 0  # After release, the note ends

def audio_callback(outdata, frames, time_info, status):
    t = (np.arange(frames) + audio_callback.frame) / SAMPLE_RATE
    out = np.zeros(frames, dtype=np.float32)

    current_time = time.time()

    with lock:
        for note, (start_time, velocity) in active_notes.items():
            # Profiler: if note just started, record latency
            if note in note_timestamps:
                latency = current_time - note_timestamps[note]
                latency_measurements.append(latency)
                print(f"[Profiler] Note {note}: {latency * 1000:.2f} ms latency")
                del note_timestamps[note]  # only once per note

            # Apply envelope to the velocity (volume)
            envelope_factor = envelope(current_time, start_time, ATTACK_TIME, DECAY_TIME, SUSTAIN_LEVEL, RELEASE_TIME)
            adjusted_velocity = velocity * envelope_factor

            freq = frequency_from_midi_note(note)
            phase = 2 * np.pi * freq * (t - start_time)
            wave = np.sin(phase) * adjusted_velocity
            out += wave

    out = np.clip(out, -1.0, 1.0)
    outdata[:] = out.reshape(-1, 1)
    audio_callback.frame += frames

audio_callback.frame = 0

# === MIDI Input Thread ===
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
                    active_notes[msg.note] = (now, velocity)
                    note_timestamps[msg.note] = now
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                with lock:
                    if msg.note in active_notes:
                        del active_notes[msg.note]
                    if msg.note in note_timestamps:
                        del note_timestamps[msg.note]

# === Main ===
if __name__ == "__main__":
    midi_thread_obj = threading.Thread(target=midi_thread, daemon=True)
    midi_thread_obj.start()

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

