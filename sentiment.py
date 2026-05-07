import numpy as np
import tempfile
import wave

from config import client, SAMPLE_RATE


def save_wav(audio_array):

    audio_array = np.array(audio_array, dtype=np.float32)

    # normalize
    audio_array = audio_array / np.max(np.abs(audio_array))

    audio_int16 = (audio_array * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        return tmp.name


def analyze_audio(recording, stop_reason=""):

    try:

        if recording is None:
            return "No speech detected", "N/A", "N/A"

        wav_file = save_wav(recording)

        # Speech-to-text
        with open(wav_file, "rb") as f:

            transcription = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3"
            )

        text = transcription.text.strip()

        if text == "":
            return "No speech detected", "N/A", "N/A"

        # Sentiment
        sentiment_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Reply only with Positive, Negative, or Neutral"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )

        sentiment = sentiment_response.choices[0].message.content.strip()

        # Emotion
        emotion_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Reply only with Joy, Sadness, Anger, Fear, or Surprise"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )

        emotion = emotion_response.choices[0].message.content.strip()

        return text, sentiment, emotion

    except Exception as e:

        return f"Error: {e}", "N/A", "N/A"