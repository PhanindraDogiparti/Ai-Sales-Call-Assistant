import numpy as np
import tempfile
import wave
import re
from config import client, SAMPLE_RATE

def _to_mono_int16(x):
    """
    Convert audio safely into mono int16 PCM
    """

    if x is None:
        return np.array([], dtype=np.int16)

    # Handle bytes input
    if isinstance(x, bytes):
        return np.frombuffer(x, dtype=np.int16)

    # Safe numpy conversion
    try:
        arr = np.array(x)
    except Exception:
        return np.array([], dtype=np.int16)

    # Empty check
    if arr.size == 0:
        return np.array([], dtype=np.int16)

    # Stereo → Mono
    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    # Convert float32
    arr = arr.astype(np.float32)

    # Normalize
    peak = np.max(np.abs(arr)) if arr.size else 0

    if peak > 1:
        arr = arr / peak

    # Convert to int16 PCM
    return np.clip(arr * 32767, -32768, 32767).astype(np.int16)


def _save_wav_int16(mono_int16):
    """
    Save numpy audio array to temporary WAV file
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:

        with wave.open(tmp_file.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(mono_int16.tobytes())

        return tmp_file.name


def _looks_like_empty_text(text):
    """
    Detect empty/invalid transcript
    """

    if not text:
        return True

    s = text.strip()

    if len(s) <= 2:
        return True

    if not re.search(r"[A-Za-z0-9]", s):
        return True

    return False


def analyze_audio(recording, stop_reason="manual"):
    """
    Main analysis function
    """

    try:

        # Silence detection
        if (
            isinstance(stop_reason, str)
            and stop_reason.lower().startswith("silent")
        ):
            return "Not Speaking", "N/A", "N/A"

        # Convert audio
        pcm = _to_mono_int16(recording)

        if pcm.size == 0:
            return "Not Speaking", "N/A", "N/A"

        # Save temp WAV
        wav_file = _save_wav_int16(pcm)

        # Speech-to-text
        with open(wav_file, "rb") as f:

            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f
            )

        text = getattr(transcription, "text", "").strip()

        # Empty transcript handling
        if _looks_like_empty_text(text):
            return "Not Speaking", "N/A", "N/A"

        # Sentiment analysis
        sentiment_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Reply with ONLY one word: "
                        "Positive, Negative, or Neutral."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )

        sentiment_label = (
            sentiment_response.choices[0]
            .message.content
            .strip()
            .split()[0]
        )

        # Emotion analysis
        emotion_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Reply with ONLY one word: "
                        "Joy, Sadness, Anger, Fear, Surprise, or Neutral."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )

        emotion_label = (
            emotion_response.choices[0]
            .message.content
            .strip()
            .split()[0]
        )

        return text, sentiment_label, emotion_label

    except Exception as e:

        return (
            f"Error: {str(e)}",
            "N/A",
            "N/A"
        )