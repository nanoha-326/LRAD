import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import re, os, json, unicodedata, base64
from datetime import datetime, timezone, timedelta
import gspread
from google.oauth2.service_account import Credentials
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import random
import time

st.set_page_config(page_title="LRADチャット", layout="centered")

# Step 1: 言語設定とサイドバーUI
lang = st.sidebar.selectbox("言語を選択 / Select Language", ["日本語", "English"], index=0)

sidebar_title = "⚙️ 設定" if lang == "日本語" else "⚙️ Settings"
font_size_label = "文字サイズを選択" if lang == "日本語" else "Select Font Size"
font_size_options = ["小", "中", "大"] if lang == "日本語" else ["Small", "Medium", "Large"]
st.sidebar.title(sidebar_title)
font_size = st.sidebar.selectbox(font_size_label, font_size_options, index=1)

font_size_map_jp = {"小": "14px", "中": "18px", "大": "24px"}
font_size_map_en = {"Small": "14px", "Medium": "18px", "Large": "24px"}
selected_font_size = font_size_map_jp[font_size] if lang == "日本語" else font_size_map_en[font_size]

st.markdown(f"""
    <style>
        div[data-testid="stVerticalBlock"] * {{ font-size: {selected_font_size}; }}
        section[data-testid="stSidebar"] * {{ font-size: {selected_font_size}; }}
    </style>
    """, unsafe_allow_html=True)

WELCOME_MESSAGES_JP = [
    "ようこそ！LRADチャットボットへ。",
    "あなたの疑問にお応えします。",
    "LRAD専用チャットボットです。",
]
WELCOME_MESSAGES_EN = [
    "Welcome to the LRAD Chat Assistant.",
    "Your questions, our answers."
]
WELCOME_MESSAGES = WELCOME_MESSAGES_JP if lang == "日本語" else WELCOME_MESSAGES_EN

LOGIN_TITLE = "ログイン" if lang == "日本語" else "Login"
LOGIN_PASSWORD_LABEL = "パスワードを入力" if lang == "日本語" else "Enter Password"
LOGIN_ERROR_MSG = "パスワードが間違っています" if lang == "日本語" else "Incorrect password"
WELCOME_CAPTION = (
    "※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。"
    if lang == "日本語" else
    "This chatbot responds based on FAQ and AI, but may not answer all questions accurately."
)
CHAT_INPUT_PLACEHOLDER = "質問をどうぞ..." if lang == "日本語" else "Ask your question..."

CORRECT_PASSWORD = "mypassword"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "welcome_shown" not in st.session_state:
    st.session_state["welcome_shown"] = False

def password_check():
    if not st.session_state["authenticated"]:
        with st.form("login_form"):
            st.title(LOGIN_TITLE)
            password = st.text_input(LOGIN_PASSWORD_LABEL, type="password")
            submitted = st.form_submit_button(LOGIN_TITLE)
            if submitted:
                if password == CORRECT_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.session_state["welcome_shown"] = False
                    st.experimental_rerun()
                else:
                    st.error(LOGIN_ERROR_MSG)
        st.stop()

password_check()

if st.session_state["authenticated"] and not st.session_state["welcome_shown"]:
    st.session_state["welcome_shown"] = True
    st.markdown(f"""
        <style>
        .fullscreen {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 48px;
            font-weight: bold;
            animation: fadein 1s ease-in-out;
            z-index: 9999;
        }}
        @keyframes fadein {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        </style>
        <div class="fullscreen">
            {random.choice(WELCOME_MESSAGES)}
        </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)
except Exception as e:
    st.error("OpenAI APIキーの取得に失敗しました。st.secretsの設定を確認してください。")
    st.stop()

# 以下、FAQ表示やチャット実装は前回を繰り返して使えます
# 意図している場合、ここから先も全文を続けます
# ご要望あれば残りの部分も全て続けて差し上げます！
