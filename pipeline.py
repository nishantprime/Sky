import sys
import time
import json
import re
import queue
import threading
import requests
import numpy as np
import sounddevice as sd
from moonshine_voice import MicTranscriber, ModelArch
from piper.voice import PiperVoice

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "lfm3:1.7b"
VOICE_MODEL_PATH = "en_US-lessac-medium.onnx"

# Shared state
session = requests.Session()
sentence_queue = queue.Queue()
is_processing = False
last_speech_ended_at = 0.0

conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a spoken voice assistant. Reply concisely in 1 to 2 spoken sentences. "
            "Never use markdown, bullet points, asterisks, or code blocks. Speak conversationally."
        ),
    }
]

print("Loading Piper TTS...")
voice = PiperVoice.load(VOICE_MODEL_PATH)


def audio_playback_worker():
    """Consumes sentence strings from the queue, synthesizes via Piper, and plays them."""
    global last_speech_ended_at

    while True:
        sentence = sentence_queue.get()
        if sentence is None:  # Sentinel value indicating LLM stream has ended
            sentence_queue.task_done()
            break

        clean_text = sentence.strip()
        if clean_text:
            # Piper stream generation and sounddevice playback
            for chunk in voice.synthesize(clean_text):
                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                sd.play(audio, chunk.sample_rate)
                sd.wait()

        sentence_queue.task_done()

    last_speech_ended_at = time.perf_counter()


def split_into_sentences(text_stream_generator):
    """
    Buffers streaming tokens and yields complete sentences.
    Avoids splitting on decimals or tiny fragments.
    """
    sentence_delimiter = re.compile(r"([.!?\n])")
    buffer = ""

    for token in text_stream_generator:
        buffer += token
        # Check if punctuation exists and the current fragment is long enough
        if len(buffer) > 20 and sentence_delimiter.search(buffer):
            parts = sentence_delimiter.split(buffer)
            # Reconstruct sentence with punctuation: [text, punct, remainder...]
            to_yield = (parts[0] + parts[1]).strip()
            buffer = "".join(parts[2:])
            if to_yield:
                yield to_yield

    # Yield remaining residue
    if buffer.strip():
        yield buffer.strip()


def stream_llm_response(user_text: str):
    """Fetches streamed tokens from Ollama and returns full response text."""
    conversation_history.append({"role": "user", "content": user_text})

    payload = {
        "model": MODEL_NAME,
        "messages": conversation_history,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": 75,
            "temperature": 0.3,
        },
    }

    try:
        response = session.post(OLLAMA_URL, json=payload, stream=True, timeout=15)
    except Exception as e:
        print(f"\n[Error querying Ollama: {e}]")
        return ""

    def token_generator():
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                print(content, end="", flush=True)
                yield content

    full_response_parts = []
    print("\nAssistant: ", end="", flush=True)

    for sentence in split_into_sentences(token_generator()):
        full_response_parts.append(sentence)
        sentence_queue.put(sentence)

    print()
    full_text = " ".join(full_response_parts)
    conversation_history.append({"role": "assistant", "content": full_text})
    return full_text


def run_turn(mic, user_text: str):
    """Handles turn lifecycle: halts mic, pipelines LLM & TTS, then restarts mic."""
    global is_processing

    # 1. Stop mic capture immediately at the hardware/PortAudio layer
    try:
        mic.stop()
    except Exception:
        pass

    # 2. Launch background audio worker
    playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
    playback_thread.start()

    # 3. Stream Ollama and push sentences to the queue
    stream_llm_response(user_text)

    # 4. Signal playback worker to shut down once queue empties and wait for it
    sentence_queue.put(None)
    playback_thread.join()

    # 5. Acoustic decay margin (absorb room reverb and speaker echo)
    time.sleep(0.2)

    # 6. Re-enable microphone capture
    try:
        mic.start()
    except Exception:
        pass

    is_processing = False


def create_callbacks(mic_holder):
    def on_partial(text):
        if not is_processing:
            print(f"\rUser: {text:<80}", end="", flush=True)

    def on_final(line):
        global is_processing
        user_text = line.text.strip()

        # Guard: Ignore empty text, ongoing turns, or delayed buffer events (<300ms post-speech)
        time_since_last_speech = time.perf_counter() - last_speech_ended_at
        if is_processing or not user_text or (time_since_last_speech < 0.3):
            return

        is_processing = True
        print(f"\rUser: {user_text}")

        # Execute turn processing in a background thread so Moonshine doesn't stall
        threading.Thread(
            target=run_turn,
            args=(mic_holder["mic"], user_text),
            daemon=True
        ).start()

    return on_partial, on_final


def main():
    print("Initializing Moonshine STT...")
    mic_holder = {}
    on_partial, on_final = create_callbacks(mic_holder)

    mic = (
        MicTranscriber()
        .language("en")
        .model_arch(ModelArch.SMALL_STREAMING)
        .on_text(on_partial)
        .on_line(on_final)
    )
    mic_holder["mic"] = mic

    try:
        mic.load()
        print("\nReady! Listening. Speak into your microphone (Ctrl+C to quit)...")
        mic.start()

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping voice assistant...")
    finally:
        try:
            mic.stop()
            mic.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
