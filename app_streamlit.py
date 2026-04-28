import time
import threading
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# from speech_to_text import calibrate_silence, record_until_silence

from sentiment import analyze_audio
from google_sheets import ensure_headers, save_to_sheets
from config import client as groq_client, sheet
from config import SAMPLE_RATE, CHANNELS, SILENCE_LIMIT, sheet, client

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="AI Speech Analysis Studio", page_icon="🎙️", layout="wide")

# ---------------- CSS ----------------
st.markdown(""" 
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div.block-container { padding-top: 2rem !important; }
.app-title { font-size: 36px; font-weight: 800; color: #111827; margin-bottom: 4px;}
.app-subtitle { color: #6B7280; margin-bottom: 20px; }
.navbar { display: inline-flex; gap: 8px; padding: 6px; background: #fff; border-radius: 14px; 
          box-shadow: 0 6px 16px rgba(17,24,39,.08); margin-top: 8px; margin-bottom: 10px; }
.navbtn { padding: 8px 14px; border-radius: 10px; border: 1px solid #E5E7EB; background: #fff; 
          color: #111827; font-weight: 600; }
.navbtn.active { background: #111827; color: #fff; border-color: #111827; }
.card { background: #fff; border: 1px solid #E5E7EB; border-radius: 16px; padding: 18px; 
        box-shadow: 0 8px 20px rgba(17,24,39,.05); }
.badge { display: inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 13px; }
.badge.pos { color:#065F46; background:#D1FAE5; }
.badge.neg { color:#991B1B; background:#FEE2E2; }
.badge.neu { color:#374151; background:#E5E7EB; }
.badge.emo { color:#111827; background:#EDE9FE; }
.transcript { border: 1px solid #E5E7EB; background: #F9FAFB; border-radius: 12px; padding: 12px; 
              min-height: 140px; color:#111827; }
.small { color:#6B7280; font-size: 13px; }
.placeholder { color:#9CA3AF; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="app-title">🎙️ AI Speech Analysis Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Real-time Speech-to-Text with Sentiment & Emotion Analysis</div>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab = st.session_state.get("tab", "Record")
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
with c1:
    if st.button("Record", key="nav_record"): tab = "Record"
with c2:
    if st.button("History", key="nav_history"): tab = "History"
with c3:
    if st.button("Analytics", key="nav_analytics"): tab = "Analytics"
with c4:
    if st.button("Purchasing History", key="nav_purchasing"): tab = "Purchasing History"
with c5:
    if st.button("Agent Summary", key="nav_agent"): tab = "Agent Summary"
st.session_state["tab"] = tab

st.markdown(
    f"""
    <div class="navbar">
      <span class="{'navbtn active' if tab=='Record' else 'navbtn'}">Record</span>
      <span class="{'navbtn active' if tab=='History' else 'navbtn'}">History</span>
      <span class="{'navbtn active' if tab=='Analytics' else 'navbtn'}">Analytics</span>
      <span class="{'navbtn active' if tab=='Purchasing History' else 'navbtn'}">Purchasing History</span>
      <span class="{'navbtn active' if tab=='Agent Summary' else 'navbtn'}">Agent Summary</span>
    </div>
    """,
    unsafe_allow_html=True
)
# ---------------- Helpers ----------------
# def _background_capture(threshold, holder, stop_event):
#     audio_list, stop_reason = record_until_silence(threshold, stop_event=stop_event)
#     holder["audio_list"] = audio_list
#     holder["stop_reason"] = stop_reason
#     holder["done"] = True



def refresh_animation(flag_key="_do_refresh"):
    if st.session_state.get(flag_key):
        with st.status("Refreshing data…", expanded=False) as s:
            ph = st.empty()
            for dots in ["", ".", "..", "...", "....", "....."]:
                ph.write(f"Updating{dots}")
                time.sleep(0.2)
            s.update(label="Data refreshed ✅")
            time.sleep(0.3)
        st.session_state[flag_key] = False
        st.rerun()

def _normalize_label(txt: str, domain: str) -> str:
    if not txt: return ""
    s = (txt or "").strip()
    first = s.split()[0].strip(",. ").title()
    if domain == "sent" and first in {"Positive", "Negative", "Neutral"}: return first
    if domain == "emo" and first in {"Joy", "Sadness", "Anger", "Fear", "Surprise"}: return first
    candidates = (["Positive","Negative","Neutral"] if domain=="sent"
                 else ["Joy","Sadness","Anger","Fear","Surprise"])
    for c in candidates:
        if c.lower() in s.lower(): return c
    return ""

def _clear_last_analysis():
    for k in ("audio", "transcript", "sentiment", "emotion", "stop_reason", "timestamp"):
        st.session_state.pop(k, None)
    st.session_state["transcript"] = None
    st.session_state["sentiment"] = "—"
    st.session_state["emotion"] = "—"

# ==== CRM CONFIG ====
CRM_SHEET_NAME = "CRM"
SUMMARIES_SHEET_NAME = "Summaries"

CRM_HEADERS = [
    "CustomerName", "Company", "Industry",
    "Budget", "InterestLevel", "Email", "Phone","RecommendedProducts"
]

SUMMARIES_HEADERS = [
    "Timestamp", "CustomerPhone",
    "Summary", "ActionItems", "Sentiment", "Emotion",
    "RecommendedProducts", "ProductPrice", "PurchaseDate"
]

PRODUCT_PRICE_MAP = {
    "CRM Suite": 12000,
    "Analytics Dashboard": 8000,
    "POS System": 7000,
    "Loyalty App": 4000,
    "Telehealth Platform": 15000,
    "Patient CRM": 9000,
    "LMS Platform": 10000,
    "Online Classrooms": 6000,
    "ERP Suite": 20000,
    "Predictive Maintenance": 12000,
    "IoT Sensors": 5000,
    "Yield Prediction AI": 7000
}

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame
    

# ---- LLM Summary generator (safe, JSON-only) ----
def generate_llm_summary(transcript: str, customer: dict, sentiment: str, emotion: str) -> tuple[str, str]:
    """Return (summary, action_items_str). If transcript is empty, return a silent-call message."""
    if not isinstance(transcript, str) or not transcript.strip():
        return ("User was not speaking. No recommendations available.", "")

    name = customer.get("CustomerName", "") if customer else ""
    industry = customer.get("Industry", "") if customer else ""

    sys = (
        "You are a sales assistant. Write a concise post-call summary and clear action items.\n"
        "- Keep summary <= 120 words.\n"
        "- Use simple bullet points in Action Items (2-4 items).\n"
        "- Avoid guessing unknown details.\n"
        "Return JSON only."
    )
    user = (
        f"Customer: {name}\n"
        f"Industry: {industry}\n"
        f"Sentiment: {sentiment}\n"
        f"Emotion: {emotion}\n"
        f"Transcript:\n{transcript}\n\n"
        "Return JSON with keys 'summary' and 'action_items' (list of strings)."
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        import json, re as _re
        data = {}
        try:
            data = json.loads(content)
        except Exception:
            m = _re.search(r"\{.*\}", content, flags=_re.S)
            if m:
                data = json.loads(m.group(0))

        summary = (str(data.get("summary", "")) or "Summary unavailable.").strip()
        items = data.get("action_items", [])
        if not isinstance(items, list):
            items = [str(items)]
        ai = "; ".join([str(x).strip() for x in items if str(x).strip()])[:400]
        return (summary, ai)
    except Exception as e:
        return (f"Summary error: {e}", "")


# ---- Save a row to the 'Summaries' sheet ----
def save_summary_row(timestamp: str,
                     customer: dict,
                     summary: str,
                     action_items: str,
                     sentiment: str,
                     emotion: str,
                     ranked_products: list):
    """
    Writes one row to the Summaries sheet. If no speech was detected, writes a clean
    fallback row with NAs and no product recommendations.
    """
    ws = ensure_summaries_ready()

    # Determine if the call had speech using your existing stop_reason + transcript
    stop_reason = (st.session_state.get("stop_reason", "") or "").lower()
    transcript_txt = (st.session_state.get("transcript", "") or "").strip()
    no_speech = (not transcript_txt) or ("no speech" in stop_reason) or stop_reason.startswith("silent")

    if no_speech:
        # Fallback (no recommendations saved)
        if not summary:
            summary = "User was not speaking. No recommendations available."
        if not action_items:
            action_items = "Try again later; Send a brief follow-up message"
        row = [
            timestamp,
            (customer or {}).get("Phone", "NA"),
            summary,
            action_items,
            "NA",
            "NA",
            "NA",
            "NA",
            "NA"
        ]
    else:
        # Normal save with products + prices (if any)
        products = ranked_products if ranked_products else []
        products_str = ", ".join(products) if products else "NA"
        prices_str = (
            ", ".join(str(PRODUCT_PRICE_MAP.get(p, "NA")) for p in products)
            if products else "NA"
        )
        row = [
            timestamp,
            (customer or {}).get("Phone", "NA"),
            summary or "NA",
            action_items or "NA",
            sentiment or "NA",
            emotion or "NA",
            products_str,
            prices_str,
            datetime.today().strftime("%Y-%m-%d"),
        ]

    ws.append_row(row)



# ---- Sheet Helpers ----
def _get_ws(title: str):
    ss = sheet.spreadsheet
    try: return ss.worksheet(title)
    except Exception: return ss.add_worksheet(title=title, rows=1000, cols=20)

def _ensure_ws_headers(ws, headers):
    values = ws.get_all_values()
    if not values:
        ws.update('A1', [headers])
    else:
        if values[0] != headers:
            ws.update(f"A1:{chr(64+len(headers))}1", [headers])

def ensure_crm_ready():
    ws = _get_ws(CRM_SHEET_NAME); _ensure_ws_headers(ws, CRM_HEADERS); return ws

def ensure_summaries_ready():
    ws = _get_ws(SUMMARIES_SHEET_NAME); _ensure_ws_headers(ws, SUMMARIES_HEADERS); return ws

@st.cache_data(ttl=300)
def load_crm_df() -> pd.DataFrame:
    ws = ensure_crm_ready()
    values = ws.get_all_values()
    if not values or len(values) < 2: return pd.DataFrame(columns=CRM_HEADERS)
    df = pd.DataFrame(values[1:], columns=values[0])
    if "Budget" in df.columns: df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce")
    return df

def get_customer_options(df: pd.DataFrame):
    labels, id_map = [], {}
    for _, row in df.iterrows():
        name, company, email = row.get("CustomerName",""), row.get("Company",""), row.get("Email","")
        label = f"{name} — {company}" if company else name
        labels.append(label); id_map[label] = email
    return labels, id_map

def get_customer_by_email(df: pd.DataFrame, email: str) -> dict:
    if not email: return {}
    hit = df.loc[df["Email"] == email]
    if hit.empty: return {}
    return hit.iloc[0].to_dict()

def parse_products(cell: str):
    if not cell: return []
    parts = [p.strip() for p in str(cell).split(",") if p.strip()]
    seen, out = set(), []
    for p in parts:
        low = p.lower()
        if low not in seen: out.append(p); seen.add(low)
    return out

def rank_products(products, sentiment: str):
    if not isinstance(sentiment, str) or not products: return []
    s = sentiment.strip().lower()
    if "neg" in s:
        soft, hard = [], []
        for p in products:
            pl = p.lower()
            if any(k in pl for k in ["trial", "demo", "lite", "basic"]): soft.append(p)
            else: hard.append(p)
        return soft + hard if soft else products
    return products
def generate_llm_summary(transcript: str, customer: dict, sentiment: str, emotion: str) -> tuple[str, str]:
    """Return (summary, action_items_str). If transcript is empty, return a silent-call message."""
    if not isinstance(transcript, str) or not transcript.strip():
        return ("Not Speaking. No summary generated.", "")

    name = customer.get("CustomerName", "") if customer else ""
    industry = customer.get("Industry", "") if customer else ""

    sys = (
        "You are a sales assistant. Write a concise post-call summary and clear action items.\n"
        "- Keep summary <= 120 words.\n"
        "- Use simple bullet points in Action Items (2-4 items).\n"
        "- Avoid guessing unknown details.\n"
        "Return JSON only."
    )
    user = (
        f"Customer: {name}\n"
        f"Industry: {industry}\n"
        f"Sentiment: {sentiment}\n"
        f"Emotion: {emotion}\n"
        f"Transcript:\n{transcript}\n\n"
        "Return JSON with keys 'summary' and 'action_items' (list of strings)."
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        import json, re as _re
        data = {}
        try:
            data = json.loads(content)
        except Exception:
            m = _re.search(r"\{.*\}", content, flags=_re.S)
            if m:
                data = json.loads(m.group(0))

        summary = str(data.get("summary", "")).strip() or "Summary unavailable."
        items = data.get("action_items", [])
        if not isinstance(items, list):
            items = [str(items)]
        ai = "; ".join([str(x) for x in items if str(x).strip()])[:400]
        return (summary, ai)
    except Exception as e:
        return (f"Summary error: {e}", "")

# ---- Objection Handling Prompts ----
def generate_objection_prompts(sentiment: str) -> list:
    if not isinstance(sentiment, str): return []
    s = sentiment.lower()
    if "neg" in s:
        return [
            "I understand your concern. Can you tell me more about what worries you?",
            "Would a trial version help you evaluate before committing?",
            "We can customize the plan to better suit your budget."
        ]
    elif "neu" in s or "neutral" in s:
        return [
            "What features are most important for your team?",
            "Would you like a quick demo to explore options?",
            "Is there any additional information you need before deciding?"
        ]
    else:  # Positive or unknown
        return [
            "Great! Would you like me to share the pricing details?",
            "Should we schedule a follow-up to finalize?",
            "Can I connect you with our support team for onboarding?"
        ]

# ---------------- RECORD TAB ----------------
if tab == "Record":
    left, right = st.columns(2)

    # --- Left: Recorder + CRM ---
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Voice Recorder")
        st.caption("Toggle Start/Stop. Stops automatically on silence, too.")
        st.write(f"**Sample Rate:** {SAMPLE_RATE/1000:.0f} kHz | **Channels:** {'Mono' if CHANNELS==1 else CHANNELS} | **Silence Limit:** {SILENCE_LIMIT}s")
        st.divider()

        # ==== CRM: Customer picker + Profile ====
        crm_df = load_crm_df()
        if crm_df.empty:
            st.info("Add some rows to the **CRM** sheet to enable real-time profile & recommendations.")
        else:
            labels, id_map = get_customer_options(crm_df)
            st.session_state["_id_map_cache"] = id_map
            current_label = st.session_state.get("selected_customer_label")
            selected_label = st.selectbox(
                "Select customer",
                options=["— Select —"] + labels,
                index=(["— Select —"] + labels).index(current_label) if current_label in (["— Select —"] + labels) else 0,
                help="Pick a customer to view profile and recommended products instantly."
            )
            st.session_state["selected_customer_label"] = selected_label

            selected_customer = {}
            if selected_label and selected_label != "— Select —":
                email_key = id_map.get(selected_label, "")
                selected_customer = get_customer_by_email(crm_df, email_key)

            if selected_customer:
                with st.expander("👤 Customer Profile", expanded=True):
                    colA, colB, colC = st.columns(3)
                    colA.metric("Name", selected_customer.get("CustomerName","—"))
                    colB.metric("Industry", selected_customer.get("Industry","—"))
                    colC.metric("Budget", str(selected_customer.get("Budget","—")))
                    colA.metric("Interest", selected_customer.get("InterestLevel","—"))
                    colB.metric("Email", selected_customer.get("Email","—"))
                    colC.metric("Phone", selected_customer.get("Phone","—"))

                    # --- Recommended Products (AFTER call with speech) ---
                    ranked = st.session_state.get("ranked_products", [])
                    if st.session_state.get("call_had_speech") and ranked:
                        st.markdown("### 🧩 Recommended Products")
                        for p in ranked:
                            price = PRODUCT_PRICE_MAP.get(p, "N/A")
                            st.write(f"- {p} — 💲 {price}")
                    elif st.session_state.get("call_had_speech") is False:
                        st.info("User was not speaking. No recommendations available.")
                    else:
                        st.caption("_Recommendations will appear after a call with detected speech._")

        # ==== Recording state ====
        st.subheader("🎙️ Live Voice Recording (WebRTC)")

        ctx = webrtc_streamer(
        key="audio",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.audio_processor:
        st.success("🎙️ Recording... Speak now")

        if st.button("🛑 Stop & Analyze", use_container_width=True):
            audio_frames = ctx.audio_processor.frames

            if len(audio_frames) > 0:
                audio_np = np.concatenate(audio_frames, axis=0)

                st.session_state["audio"] = audio_np
                st.session_state["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

                st.success("✅ Audio captured successfully")
            else:
                st.warning("⚠️ No audio recorded")
        # st.session_state.setdefault("rec_thread", None)
        # st.session_state.setdefault("rec_holder", {})
        # st.session_state.setdefault("rec_stop", None)
        # st.session_state.setdefault("rec_start_ts", None)
        # st.session_state.setdefault("is_recording", False)

        # # Toggle button
        # label = "⏹️ Stop Recording" if st.session_state.is_recording else "🔴 Start Recording"
        # toggled = st.button(label, use_container_width=True)

        # if toggled:
        #     if not st.session_state.is_recording:
        #         # reset old results
        #         for k in ("audio","transcript","sentiment","emotion","stop_reason",
        #                   "timestamp","ranked_products","call_had_speech"):
        #             st.session_state.pop(k, None)
        #         st.session_state["transcript"] = None

        #         with st.status("Calibrating baseline noise…", expanded=True) as s:
        #             thr = calibrate_silence()
        #             s.write(f"Calibrated threshold = {thr:.6f}")
        #             s.update(label="Listening… Speak now.")

        #         holder = {"done": False}
        #         stop_event = threading.Event()
        #         t = threading.Thread(target=_background_capture, args=(thr, holder, stop_event), daemon=True)
        #         t.start()

        #         st.session_state.rec_holder = holder
        #         st.session_state.rec_stop = stop_event
        #         st.session_state.rec_thread = t
        #         st.session_state.rec_start_ts = time.time()
        #         st.session_state.is_recording = True
        #     else:
        #         if st.session_state.rec_stop is not None:
        #             st.session_state.rec_stop.set()

        # # Timer while recording
        # if st.session_state.is_recording and st.session_state.rec_thread and st.session_state.rec_thread.is_alive():
        #     elapsed = int(time.time() - (st.session_state.rec_start_ts or time.time()))
        #     st.markdown(f"**⏱️ Recording:** {elapsed:02d} sec")
        #     time.sleep(1)
        #     st.rerun()

        # After recording stops
        if st.session_state.rec_thread is not None and not st.session_state.rec_thread.is_alive():
            holder = st.session_state.rec_holder or {}
            if holder.get("done") and "audio" not in st.session_state:
                audio_list = holder.get("audio_list") or []
                stop_reason = holder.get("stop_reason", "")
                st.session_state["stop_reason"] = stop_reason

                if stop_reason.lower().startswith("silent") and len(audio_list) > SILENCE_LIMIT:
                    audio_list = audio_list[:len(audio_list)-SILENCE_LIMIT]

                if audio_list:
                    merged = np.concatenate(audio_list, axis=0)
                    st.session_state["audio"] = merged
                    st.session_state["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.toast(f"Captured {merged.shape[0]/SAMPLE_RATE:.1f} sec", icon="🎧")
                else:
                    st.warning("No audio captured.")

            # reset state
            st.session_state.rec_thread = None
            st.session_state.rec_stop = None
            st.session_state.rec_holder = {}
            st.session_state.rec_start_ts = None
            st.session_state.is_recording = False
            st.rerun()

        # Auto-analyze
        if "audio" in st.session_state and st.session_state.get("transcript") is None:
            with st.spinner("Analyzing…"):
                transcript, sentiment_label, emotion_label = analyze_audio(
                    st.session_state["audio"],
                    st.session_state.get("stop_reason","")
                )
            st.session_state["transcript"] = transcript
            st.session_state["sentiment"] = sentiment_label
            st.session_state["emotion"] = emotion_label

            # Detect if speech happened
            stop_reason = (st.session_state.get("stop_reason","") or "").lower()
            call_had_speech = bool(transcript and transcript.strip()) and not (
                "no speech" in stop_reason or stop_reason.startswith("silent")
            )
            st.session_state["call_had_speech"] = call_had_speech

            # compute recommendations only when speech detected & customer selected
            if call_had_speech and selected_customer:
                base_products = parse_products(selected_customer.get("RecommendedProducts",""))
                ranked = rank_products(base_products, sentiment_label)
                st.session_state["ranked_products"] = ranked
            else:
                st.session_state["ranked_products"] = []

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Right: Results ---
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Analysis Results")
        st.caption("Transcription and sentiment analysis")

        st.markdown("**Transcript**")
        transcript_text = st.session_state.get("transcript", None)
        if transcript_text is None:
            st.markdown('<div class="transcript"><span class="placeholder">Waiting for new recording...</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="transcript">{transcript_text}</div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            st.markdown("**Sentiment**")
            sent = st.session_state.get("sentiment","—")
            css = "neu"
            if isinstance(sent,str):
                s = sent.strip().lower()
                if "pos" in s: css = "pos"
                elif "neg" in s: css = "neg"
            st.markdown(f'<span class="badge {css}">{sent}</span>', unsafe_allow_html=True)
        with b:
            st.markdown("**Emotion**")
            emo = st.session_state.get("emotion","—")
            st.markdown(f'<span class="badge emo">{emo}</span>', unsafe_allow_html=True)

        # Suggested prompts
        st.markdown("**Suggested Objection Handling Prompts**")
        prompts = generate_objection_prompts(st.session_state.get("sentiment",""))
        if prompts:
            for p in prompts:
                st.write(f"• {p}")
        else:
            st.write("No suggestions available.")

        # Save to Sheets
        save_summary_too = st.checkbox("Also save post-call summary to 'Summaries'", value=True)
        if st.button("💾 Save to Google Sheets", use_container_width=True):
            ts = st.session_state.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            try:
                ensure_headers()
                save_to_sheets(
                    ts,
                    st.session_state.get("transcript",""),
                    st.session_state.get("sentiment",""),
                    st.session_state.get("emotion",""),
                    st.session_state.get("stop_reason","")
                )
                st.success("Saved to Google Sheets.")

                if save_summary_too:
                    transcript_val = st.session_state.get("transcript","").strip()
                    sentiment_val = st.session_state.get("sentiment","")
                    emotion_val = st.session_state.get("emotion","")

                    if st.session_state.get("call_had_speech"):
                        summary, action_items = generate_llm_summary(transcript_val, selected_customer, sentiment_val, emotion_val)
                    else:
                        summary, action_items = ("User was not speaking. No recommendations available.", "")

                    try:
                        save_summary_row(ts, selected_customer, summary, action_items, sentiment_val, emotion_val, st.session_state.get("ranked_products", []))
                        st.toast("Summary saved to 'Summaries' ✅", icon="📝")
                    except Exception as e:
                        st.warning(f"Summary save skipped: {e}")
            except Exception as e:
                st.error(f"Save failed: {e}")

        stop_reason = st.session_state.get("stop_reason","")
        if stop_reason:
            st.markdown(f'<div class="small">Stop Reason: <b>{stop_reason}</b></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HISTORY TAB ----------------
if tab == "History":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📜 Call History")
    st.caption("All saved calls with transcripts, sentiment, and emotion.")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("↻ Refresh", use_container_width=True):
            st.session_state["_refresh_history"] = True
            st.rerun()

    # Small refresh animation
    if st.session_state.get("_refresh_history"):
        with st.status("Refreshing call history…", expanded=False) as s:
            for dots in ["", ".", "..", "..."]:
                s.update(label=f"Updating{dots}")
                time.sleep(0.3)
            s.update(label="✅ Data refreshed")
        st.session_state["_refresh_history"] = False
        st.rerun()

    try:
        # Pull all values from main sheet
        values = sheet.get_all_values()
        headers = values[0] if values else []
        rows = values[1:] if values and len(values) > 1 else []

        if rows:
            import pandas as pd
            df = pd.DataFrame(rows, columns=headers)

            # Only keep key columns if available
            key_cols = [c for c in ["Timestamp", "Transcript", "Sentiment", "Emotion", "StopReason"] if c in df.columns]
            df = df[key_cols] if key_cols else df

            # Display table
            st.dataframe(
                df,
                use_container_width=True,
                height=400,
            )

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "call_history.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No call history found yet. Record and save a call to see data here.")
    except Exception as e:
        st.error(f"Error loading call history: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ANALYTICS TAB ----------------
if tab == "Analytics":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analytics")
    

    col_refresh, _ = st.columns([1, 6])
    with col_refresh:
        if st.button("↻ Refresh data", use_container_width=False):
            st.session_state["_do_refresh"] = True
            st.rerun()
    refresh_animation("_do_refresh")

    try:
        values = sheet.get_all_values()
        headers = values[0] if values else []
        rows = values[1:] if values and len(values) > 1 else []

        def col_idx(name):
            try: return headers.index(name)
            except ValueError: return None

        i_sent = col_idx("Sentiment")
        i_emo  = col_idx("Emotion")

        total = len(rows)
        st.metric("Total Recordings", total)

        from collections import Counter
        import pandas as pd

        c1, c2 = st.columns(2)

        # --- Sentiment (raw) ---
        with c1:
            st.markdown("**Sentiment Distribution**")
            if i_sent is not None:
                sentiments = [r[i_sent].strip() for r in rows if i_sent < len(r) and r[i_sent].strip()]
                counts = Counter(sentiments)
                if counts:
                    df = pd.DataFrame(list(counts.items()), columns=["Sentiment", "Count"]).sort_values("Count", ascending=False)
                    st.bar_chart(df.set_index("Sentiment")["Count"], height=260)
                else:
                    st.info("No sentiment data yet.")
            else:
                st.warning("No Sentiment column in sheet.")

        # --- Emotion (raw) ---
        with c2:
            st.markdown("**Emotion Distribution**")
            if i_emo is not None:
                emotions = [r[i_emo].strip() for r in rows if i_emo < len(r) and r[i_emo].strip()]
                counts = Counter(emotions)
                if counts:
                    df = pd.DataFrame(list(counts.items()), columns=["Emotion", "Count"]).sort_values("Count", ascending=False)
                    st.bar_chart(df.set_index("Emotion")["Count"], height=260)
                else:
                    st.info("No emotion data yet.")
            else:
                st.warning("No Emotion column in sheet.")

    except Exception as e:
        st.error(f"Analytics error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PURCHASING HISTORY TAB ----------------
elif tab == "Purchasing History":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🛒 Purchasing History")

    email_input = st.text_input("Enter Customer Email", key="purchase_email")

    if email_input:
        try:
            ws = ensure_summaries_ready()
            values = ws.get_all_values()
            headers = values[0] if values else []
            rows = values[1:] if len(values) > 1 else []

            crm_df = load_crm_df()
            customer = get_customer_by_email(crm_df, email_input)
            phone_number = customer.get("Phone") if customer else None

            matched = []
            if phone_number:
                phone_idx = headers.index("CustomerPhone") if "CustomerPhone" in headers else None
                if phone_idx is not None:
                    for r in rows:
                        if phone_idx < len(r) and r[phone_idx] == phone_number:
                            matched.append({headers[i]: (r[i] if i < len(r) else "") for i in range(len(headers))})

            if matched:
                st.success(f"Found {len(matched)} past purchases for {email_input}")

                # ---- Total Spend ----
                total_spend = 0
                for m in matched:
                    if "ProductPrice" in m and m["ProductPrice"]:
                        try:
                            total_spend += sum([
                                float(x)
                                for x in m["ProductPrice"].split(",")
                                if x.strip().replace('.', '', 1).isdigit()
                            ])
                        except:
                            pass
                st.metric("Total Spend", f"${total_spend:,.2f}")

                # ✅ Convert to DataFrame and remove Summary + ActionItems
                df_display = pd.DataFrame(matched)
                cols_to_drop = [c for c in ["Summary", "ActionItems"] if c in df_display.columns]
                df_display = df_display.drop(columns=cols_to_drop)

                # 📊 Show cleaned table
                st.dataframe(df_display, use_container_width=True, height=350)
            else:
                st.info("No purchase history found for this customer.")

            # --- Recommendations ---
            if customer:
                already_bought = []
                for m in matched:
                    if "RecommendedProducts" in m and m["RecommendedProducts"]:
                        already_bought += parse_products(m["RecommendedProducts"])
                already_bought = set(already_bought)

                base_products = parse_products(customer.get("RecommendedProducts", ""))
                current_sentiment = st.session_state.get("sentiment", "")
                ranked = rank_products(base_products, current_sentiment)

                recos = [p for p in ranked if p not in already_bought]

                st.markdown("### 🎯 Recommended Next Products")
                if recos:
                    cols = st.columns(len(recos))
                    for idx, p in enumerate(recos):
                        price = PRODUCT_PRICE_MAP.get(p, "N/A")
                        price_str = f"${float(price):,.2f}" if str(price).replace('.', '', 1).isdigit() else price
                        with cols[idx]:
                            st.markdown(
                                f"""
                                <div style="
                                    background: #fff;
                                    border: 1px solid #e5e7eb;
                                    border-radius: 12px;
                                    padding: 14px;
                                    text-align: center;
                                    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                                ">
                                    <div style="font-size:16px; font-weight:600; margin-bottom:6px;">
                                        {p}
                                    </div>
                                    <div style="color:#10b981; font-weight:700; margin-bottom:8px;">
                                        💲 {price_str}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    # --- AI fallback with only ProductName + Price ---
                    try:
                        prompt = f"""
                        You are a sales assistant. The customer profile is:
                        Name: {customer.get("CustomerName")}
                        Industry: {customer.get("Industry")}
                        Budget: {customer.get("Budget")}
                        Interest: {customer.get("InterestLevel")}
                        Sentiment: {current_sentiment}

                        Suggest 3 new products that would be valuable for this customer.
                        Return ONLY a JSON array like:
                        [
                          {{"ProductName": "CRM Suite", "Price": 12000}},
                          {{"ProductName": "Analytics Dashboard", "Price": 8000}},
                          {{"ProductName": "POS System", "Price": 7000}}
                        ]
                        """
                        resp = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are a helpful sales assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.4
                        )
                        import json, re
                        content = resp.choices[0].message.content.strip()

                        # ✅ Extract only valid JSON array
                        match = re.search(r"\[.*\]", content, flags=re.S)
                        products_ai = json.loads(match.group(0)) if match else []

                        if products_ai:
                            st.info("🤖 AI-Suggested Products:")
                            cols = st.columns(len(products_ai))
                            for idx, prod in enumerate(products_ai):
                                name = prod.get("ProductName", "Unnamed")
                                price = prod.get("Price", "N/A")
                                price_str = f"${float(price):,.2f}" if str(price).replace('.', '', 1).isdigit() else price
                                with cols[idx]:
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background: #fff;
                                            border: 1px solid #e5e7eb;
                                            border-radius: 12px;
                                            padding: 14px;
                                            text-align: center;
                                            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                                        ">
                                            <div style="font-size:16px; font-weight:600; margin-bottom:6px;">
                                                {name}
                                            </div>
                                            <div style="color:#10b981; font-weight:700; margin-bottom:8px;">
                                                💲 {price_str}
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.warning("No recommendations available from CRM or AI.")
                    except Exception as e:
                        st.error(f"AI recommendation error: {e}")

        except Exception as e:
            st.error(f"Error fetching history: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- AGENT SUMMARY TAB ----------------
elif tab == "Agent Summary":
    import textwrap
    from groq import Groq

    # 🧠 Auto logout if user switches tab
    if "last_tab" in st.session_state and st.session_state["last_tab"] != tab:
        if st.session_state["last_tab"] == "Agent Summary":
            st.session_state["agent_logged_in"] = False  # logout only when leaving agent tab
    st.session_state["last_tab"] = tab

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Post-Call Summaries (Agent Only)")
    st.caption("Private view for agents. Enter customer phone number to view their summaries.")

    # ✅ Simple agent login
    if not st.session_state.get("agent_logged_in", False):
        with st.form("login_form"):
            username = st.text_input("👤 Agent Username")
            password = st.text_input("🔑 Password", type="password")
            submit_login = st.form_submit_button("Login")

            if submit_login:
                if username == "agent" and password == "1234":
                    st.session_state["agent_logged_in"] = True
                    st.success("✅ Logged in successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        st.stop()
    else:
        st.success("Welcome Agent 👋")

    # 🚪 Manual Logout
    if st.button("🚪 Logout"):
        st.session_state["agent_logged_in"] = False
        st.rerun()

    # 🔍 Input for phone number
    customer_filter = st.text_input("📞 Enter Customer Phone Number")

    if not customer_filter.strip():
        st.info("Enter phone number to view summaries.")
        st.stop()

    try:
        # ✅ Load data from Summaries sheet
        ws = ensure_summaries_ready()
        values = ws.get_all_values()
        if not values or len(values) < 2:
            st.warning("No data found in Summaries sheet.")
            st.stop()

        headers = values[0]
        rows = values[1:]
        df = pd.DataFrame(rows, columns=headers)

        if "CustomerPhone" not in df.columns:
            st.error("❌ 'CustomerPhone' column not found in sheet.")
            st.stop()

        # 🔍 Filter by phone
        mask = df["CustomerPhone"].astype(str).str.contains(customer_filter, case=False, na=False, regex=False)
        filtered = df[mask]

        if filtered.empty:
            st.warning(f"No summaries found for **{customer_filter}**.")
        else:
            st.info(f"📋 Showing {len(filtered)} summaries for **{customer_filter}**")

            # 🧠 Collect all summaries
            all_summaries_text = ""
            for _, row in filtered.iterrows():
                st.markdown(
                    f"""
                    <div style="
                        background:#fff;
                        border:1px solid #E5E7EB;
                        border-radius:14px;
                        padding:16px 20px;
                        margin-bottom:14px;
                        box-shadow:0 4px 10px rgba(0,0,0,0.05);
                    ">
                        <h4 style="margin:0; color:#111827;">🕒 {row.get('Timestamp', '')}</h4>
                        <p style="color:#6B7280; margin:2px 0 8px;">📞 <b>{row.get('CustomerPhone','NA')}</b></p>
                        <p><b>📝 Summary:</b> {row.get('Summary','')}</p>
                        <p><b>🎯 Action Items:</b> {row.get('ActionItems','')}</p>
                        <p><b>😃 Sentiment:</b> {row.get('Sentiment','N/A')}</p>
                        <p><b>💭 Emotion:</b> {row.get('Emotion','N/A')}</p>
                        <p><b>🧩 Recommended Products:</b> {row.get('RecommendedProducts','N/A')}</p>
                        <p><b>💰 Prices:</b> {row.get('ProductPrice','N/A')}</p>
                        <p style="color:#6B7280;">📅 Purchase Date: {row.get('PurchaseDate','')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                all_summaries_text += f"Summary: {row.get('Summary','')}\nAction Items: {row.get('ActionItems','')}\n\n"

            # 🧠 AI Summary Button
            if st.button("🤖 Generate AI Summary"):
                try:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    with st.spinner("Generating AI summary... ⏳"):
                        prompt = f"""
                        You are an assistant summarizing multiple call summaries into a structured post-call report.
                        Combine all details below into formatted sections:
                        💬 Overall Sentiment**
                        🎯 Customer Intent
                        🧩 Key Topics
                        ⚠️ Objections
                        ✅ Resolutions
                        📝 Next Steps
                        🔁 Recommended Follow-up

                        Input Summaries:
                        {all_summaries_text}
                        """

                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                        )

                        ai_summary = response.choices[0].message.content.strip()

                    st.success("✅ AI Summary Generated")
                    st.markdown(
                        f"""
                        <div style="background:#F9FAFB; border:1px solid #E5E7EB; border-radius:10px; padding:16px;">
                        {ai_summary.replace("\n", "<br>")}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"AI Summary generation failed: {e}")

    except Exception as e:
        st.error(f"⚠️ Error loading summaries: {e}")

    st.markdown('</div>', unsafe_allow_html=True)






