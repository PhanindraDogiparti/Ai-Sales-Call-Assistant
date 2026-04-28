# 🎤🤖 AI Sales Call Assistant

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🚀 **Live Demo:** *(Add your deployed link here)*

---

## 📌 Overview

AI Sales Call Assistant is a **real-time AI-powered application** that helps sales agents **record, transcribe, analyze, and summarize calls automatically**.

It supports both:

* 🖥️ **Local microphone recording (sounddevice)**
* 🌐 **Browser-based recording (WebRTC)** for cloud deployment

The system uses **Groq Whisper** for transcription and **Groq LLaMA3** for sentiment, emotion detection, and intelligent summaries.

---

## 📸 Application Screenshots

### 🎙️ Voice Recording Interface

![Voice Recorder](assets\Homapage.png)

---

### 📊 Call History

![Call History](assets\Historypage.png)

---

### 📈 Analytics Dashboard

![Analytics](assets\analyticspage.png)

---

### 🛒 Purchasing History

![Purchasing](assets\purchasinghistorypage.png)

---

### 🔐 Agent Login

![Login](assets\agentsummarypage.png)

---

### 🧠 Post-Call Summary View

![Summary](assets\postcallsummarypage.png)

---

### 🤖 AI Generated Report

![AI Report](assets\Aigenratedsummary.png)

---

## ✨ Features

### 🎧 Voice Processing

* 🎙️ Real-time recording (Local + WebRTC)
* ✍️ Speech-to-text using Whisper
* 💬 Sentiment Analysis (Positive / Neutral / Negative)
* 🎭 Emotion Detection

### 📊 Data Management

* 📁 Call history tracking
* 📈 Analytics dashboard (visual insights)
* 📤 CSV export
* 📊 Google Sheets integration

### 🧠 AI Intelligence

* 🤖 AI-generated post-call summaries
* 🎯 Customer intent detection
* ⚠️ Objection handling suggestions
* 🔁 Follow-up recommendations

### 🔐 Security

* Agent login system
* Session-based access control

---

## 🧠 Tech Stack

| Category   | Technology            |
| ---------- | --------------------- |
| Language   | Python 3.11           |
| UI         | Streamlit             |
| Voice      | sounddevice + WebRTC  |
| AI         | Groq Whisper + LLaMA3 |
| Storage    | Google Sheets         |
| Data       | pandas                |
| Deployment | Streamlit Cloud       |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ai-sales-call-assistant.git
cd ai-sales-call-assistant
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Configuration

### Groq API

```bash
GROQ_API_KEY=your_api_key_here
```

### Streamlit Secrets

```toml
GROQ_API_KEY = "your_key"
GOOGLE_CREDENTIALS = '{"type":"service_account",...}'
```

---

## ▶️ Usage

### 🖥️ Local Mode

```bash
python groq_assistant.py
```

✔ Uses system mic (`sounddevice`)

---

### 🌐 Web Mode

```bash
streamlit run app_streamlit.py
```

✔ Uses browser mic (WebRTC)

---

## 🌐 Deployment

1. Push to GitHub
2. Go to Streamlit Cloud
3. Select repo
4. Add secrets
5. Deploy

---

## ⚠️ Notes

* sounddevice works only locally
* WebRTC is used for deployed apps
* Never expose API keys

---

## 🔐 Demo Login

```
Username: agent  
Password: 1234
```

---

## 🚀 Future Improvements

* CRM Integration
* Real-time call streaming
* Multi-language support
* Advanced analytics

---

## 📄 License

MIT License

---

## 🙌 Author

**Phanindra (Phani)**
AI & ML Enthusiast 🚀
