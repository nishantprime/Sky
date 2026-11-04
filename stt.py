
"""Low-latency live microphone transcription using Moonshine."""

import time
from moonshine_voice import MicTranscriber, ModelArch


def on_partial(text):
    # Continuously updated transcription
    print(f"\r{text:<100}", end="", flush=True)


def on_final(line):
    # Finalized text
    print(f"\rFINAL: {line.text}")


def main():
    print("Initializing Moonshine...")

    mic = (
        MicTranscriber()
        .language("en")
        .model_arch(ModelArch.SMALL_STREAMING)
        # .model_arch(ModelArch.TINY/SMALL_STREAMING)
        .on_text(on_partial)
        .on_line(on_final)
    )

    try:
        mic.load()
        print("\nListening. Press Ctrl+C to stop.")
        mic.start()

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        mic.stop()
        mic.close()


if __name__ == "__main__":
    main()
