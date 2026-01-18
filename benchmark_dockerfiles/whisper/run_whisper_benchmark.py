import time
from pathlib import Path

import whisper
import whisper.audio as wa  # for model constants

AUDIO_PATH = Path("/app/audio/input.ogg")


def main() -> None:
    """
    Run the benchmark.
    :return: None
    """

    # Get the internal representation of the number of frames in the input audio.
    mel = whisper.log_mel_spectrogram(whisper.load_audio(AUDIO_PATH))  # shape: (80, frames)
    num_frames = mel.shape[1]

    model = whisper.load_model("medium", device="cuda")

    # Run transcription
    start = time.perf_counter()
    _output = model.transcribe(
        str(AUDIO_PATH),
        language="en",
        task="transcribe",
        fp16=True,
        verbose=None,
        temperature=0.0,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )
    elapsed = time.perf_counter() - start

    frames_per_second = num_frames / elapsed

    print(frames_per_second)


if __name__ == "__main__":
    main()
