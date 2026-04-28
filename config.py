# import os
# from groq import Groq
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# # ✅ Groq setup
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # 🎤 Audio settings
# SAMPLE_RATE = 16000
# CHANNELS = 1
# SILENCE_LIMIT = 5
# CSV_FILE = "groq_transcripts.csv"

# # 🔹 Google Sheets setup
# scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
# creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
# client_gs = gspread.authorize(creds)
# sheet = client_gs.open("AI Sales Call Assistant").sheet1



import os
import json
import streamlit as st
from groq import Groq
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ✅ Groq setup
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 🎤 Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_LIMIT = 5
CSV_FILE = "groq_transcripts.csv"

# 🔹 Google Sheets setup
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])

creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client_gs = gspread.authorize(creds)
sheet = client_gs.open("AI Sales Call Assistant").sheet1

