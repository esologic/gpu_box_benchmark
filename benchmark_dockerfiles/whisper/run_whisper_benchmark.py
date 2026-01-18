import json
import subprocess
import time
from pathlib import Path

import whisper

AUDIO_PATH = Path("/app/audio/input.ogg")


def probe_audio(path: Path) -> tuple[float, int]:
    """
    Returns (duration_seconds, sample_rate) from ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    data = json.loads(result.stdout)

    duration = float(data["format"]["duration"])
    sample_rate = int(data["streams"][0]["sample_rate"])

    return duration, sample_rate


def main():
    # Probe audio file
    audio_duration, sample_rate = probe_audio(AUDIO_PATH)

    # Load model (full-fat)
    model = whisper.load_model("medium", device="cuda")

    start = time.perf_counter()

    output = model.transcribe(
        str(AUDIO_PATH),
        language="en",
        task="transcribe",
        fp16=True,
        verbose=False,
        temperature=0.0,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )

    print(output)

    elapsed = time.perf_counter() - start

    metric = elapsed / audio_duration

    print(metric)


if __name__ == "__main__":
    main()
