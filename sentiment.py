import numpy as np
import tempfile
import wave
import re
from config import client, SAMPLE_RATE, CHANNELS

# def _to_mono_int16(x: np.ndarray) -> np.ndarray:
#     """Ensure (N,) mono int16 PCM from float arrays (N,), (N,1), or (N,C)."""
#     if x is None or len(x) == 0:
#         return np.array([], dtype=np.int16)
#     arr = np.asarray(x)
#     if arr.ndim == 2:
#         arr = arr.mean(axis=1)  # downmix to mono
#     arr = arr.astype(np.float32, copy=False)
#     peak = np.max(np.abs(arr)) if arr.size else 0.0
#     if peak > 1.0:
#         arr = arr / peak
#     return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)

def _to_mono_int16(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=np.int16)
    if isinstance(x, list):                          # ← ADD THIS
        x = np.concatenate(x, axis=-1)              # ← ADD THIS
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.int16)
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            arr = arr.mean(axis=0)  # WebRTC: (C, N)
        else:
            arr = arr.mean(axis=1)  # Old mic: (N, C)
    peak = np.max(np.abs(arr)) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)

def _save_wav_int16(mono_int16: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        with wave.open(tmp_file.name, 'wb') as wf:
            wf.setnchannels(1)       # mono
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(mono_int16.tobytes())
        return tmp_file.name

def _looks_like_empty_text(t: str) -> bool:
    """Treat as empty if blank/only punctuation/<=2 chars without letters/digits."""
    if not t:
        return True
    s = t.strip()
    if not s or len(s) <= 2:
        return True
    if not re.search(r"[A-Za-z0-9]", s):
        return True
    return False

def analyze_audio(recording, stop_reason: str):
    """
    If stop_reason indicates silence → return ('Not Speaking','N/A','N/A').
    Otherwise transcribe and classify normally.
    """
    # ✅ Your requested rule:
    if isinstance(stop_reason, str) and stop_reason.lower().startswith("silent"):
        return "Not Speaking", "N/A", "N/A"

    # Convert audio to mono int16
    pcm = _to_mono_int16(recording)
    if pcm.size == 0:
        return "Not Speaking", "N/A", "N/A"

    # Write temp WAV
    wav_file = _save_wav_int16(pcm)

    # Transcribe
    try:
        with open(wav_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f
            )
        text = (getattr(transcription, "text", "") or "").strip()
    except Exception as e:
        return f"[STT error: {e}]", "N/A", "N/A"

    # If transcript is essentially empty, treat as no speech
    if _looks_like_empty_text(text):
        return "Not Speaking", "N/A", "N/A"

    # Sentiment
    try:
        sentiment = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Reply with only one word: Positive, Negative, or Neutral."},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )
        sentiment_result = (sentiment.choices[0].message.content or "").strip().split()[0]
    except Exception as e:
        sentiment_result = f"Error:{e}"

    # Emotion
    try:
        emotion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Reply with only one word: Joy, Sadness, Anger, Fear, or Surprise."},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )
        emotion_result = (emotion.choices[0].message.content or "").strip().split()[0]
    except Exception as e:
        emotion_result = f"Error:{e}"

    return text, sentiment_result, emotion_result
