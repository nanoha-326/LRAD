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
    .login-outer {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        width: 100vw;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
        background-color: white;
    }}
    .login-container {{
        text-align: left;
        max-width: 400px;
        width: 100%;
        padding: 2em;
    }}
    .login-title {{
        font-size: 2.5em;
        margin-bottom: 1em;
        text-align: center;
    }}
    .login-button button {{
        background-color: #333 !important;
        color: white !important;
        border: none;
        padding: 0.5em 2em;
        font-size: 1.2em;
        border-radius: 4px;
        margin-top: 1em;
        width: 100%;
    }}
    input[type="password"] {{
        font-size: 1.2em;
        padding: 0.5em;
        width: 100%;
        max-width: 300px;
    }}
</style>
""", unsafe_allow_html=True)

WELCOME_MESSAGES = [
    "ようこそ！LRADチャットボットへ。",
    "あなたの疑問にお応えします。",
    "LRAD専用チャットボットです。"
] if lang == "日本語" else [
    "Welcome to the LRAD Chat Assistant.",
    "Your questions, our answers."
]

LOGIN_TITLE = "LRADチャットへログイン" if lang == "日本語" else "Login to LRAD Chat"
LOGIN_PASSWORD_LABEL = "パスワードを入力してください" if lang == "日本語" else "Please enter password"
LOGIN_ERROR_MSG = "パスワードが間違っています" if lang == "日本語" else "Incorrect password"
WELCOME_CAPTION = "※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。" if lang == "日本語" else "This chatbot responds based on FAQ and AI, but may not answer all questions accurately."
CHAT_INPUT_PLACEHOLDER = "質問をどうぞ..." if lang == "日本語" else "Ask your question..."
CORRECT_PASSWORD = "mypassword"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "show_welcome" not in st.session_state:
    st.session_state["show_welcome"] = False
if "welcome_message" not in st.session_state:
    st.session_state["welcome_message"] = ""
if "fade_out" not in st.session_state:
    st.session_state["fade_out"] = False

def password_check():
    if not st.session_state["authenticated"]:
        st.markdown('<div class="login-outer">', unsafe_allow_html=True)
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="login-title">{LOGIN_TITLE}</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            password = st.text_input("", type="password", placeholder=LOGIN_PASSWORD_LABEL, label_visibility="collapsed")
            submitted = st.form_submit_button("ログイン")
            if submitted:
                if password == CORRECT_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.session_state["show_welcome"] = True
                    st.session_state["welcome_message"] = random.choice(WELCOME_MESSAGES)
                    st.session_state["fade_out"] = False
                    st.experimental_rerun()
                else:
                    st.error(LOGIN_ERROR_MSG)
        st.markdown('</div></div>', unsafe_allow_html=True)
        st.stop()

password_check()

def show_welcome_screen():
    st.markdown(
        f"""
        <style>
        .fullscreen {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 56px;
            font-weight: bold;
            animation: fadein 1.5s forwards;
            z-index: 9999;
            text-align: center;
            padding: 0 20px;
            word-break: break-word;
        }}
        .fadeout {{ animation: fadeout 1.5s forwards; }}
        @keyframes fadein {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        @keyframes fadeout {{ from {{ opacity: 1; }} to {{ opacity: 0; }} }}
        </style>
        <div class="fullscreen {'fadeout' if st.session_state['fade_out'] else ''}">
            {st.session_state['welcome_message']}
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state["show_welcome"]:
    show_welcome_screen()
    if not st.session_state["fade_out"]:
        time.sleep(2)
        st.session_state["fade_out"] = True
        st.experimental_rerun()
    else:
        time.sleep(1)
        st.session_state["show_welcome"] = False
        st.experimental_rerun()

# 続きのチャット処理などはここに記述
