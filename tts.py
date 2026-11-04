import time
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

voice = PiperVoice.load("en_US-lessac-medium.onnx")

text = "Hello! I am running completely locally on your Asus Vivobook."

start = time.perf_counter()
first_chunk = True

for chunk in voice.synthesize(text):
    if first_chunk:
        ttfa = (time.perf_counter() - start) * 1000

        print(f"Piper CPU Time To First Audio: {ttfa:.1f} ms")
        first_chunk = False

    # Convert Piper's int16 audio bytes into a NumPy array
    audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

    # Play the audio
    sd.play(audio, chunk.sample_rate)
    sd.wait()

if first_chunk:
    print("No audio chunks generated.")
else:
    print("Playback complete.")
