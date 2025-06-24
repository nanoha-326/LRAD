# LRADサポートチャット（管理者認証＆Insights非表示対応）
import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, random, re, unicodedata, json, base64
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone, timedelta
import time
import traceback

st.set_page_config(page_title="LRADチャット", layout="centered")

# --- 管理者認証部分 --- #
CORRECT_PASSWORD = "mypassword"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "show_welcome" not in st.session_state:
    st.session_state["show_welcome"] = False
if "welcome_message" not in st.session_state:
    st.session_state["welcome_message"] = ""
if "fade_out" not in st.session_state:
    st.session_state["fade_out"] = False

WELCOME_MESSAGES = [
    "ようこそ！LRADチャットボットへ。",
    "あなたの疑問にお応えします。",
    "LRAD専用チャットボットです。",
    "Welcome to the LRAD Chat Assistant.",
    "Your questions, our answers.",
]

def password_check():
    if not st.session_state["authenticated"]:
        with st.form("login_form"):
            st.title("ログイン")
            password = st.text_input("パスワードを入力", type="password")
            submitted = st.form_submit_button("ログイン")
            if submitted:
                if password == CORRECT_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.session_state["is_admin"] = True
                    st.session_state["show_welcome"] = True
                    st.session_state["welcome_message"] = random.choice(WELCOME_MESSAGES)
                    st.session_state["fade_out"] = False
                    st.experimental_rerun()
                else:
                    st.error("パスワードが間違っています")
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
            font-size: 48px;
            font-weight: bold;
            animation: fadein 1.5s forwards;
            z-index: 9999;
        }}
        .fadeout {{
            animation: fadeout 1.5s forwards;
        }}
        @keyframes fadein {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        @keyframes fadeout {{
            from {{ opacity: 1; }}
            to {{ opacity: 0; }}
        }}
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
        time.sleep(2.0)
        st.session_state["fade_out"] = True
        st.experimental_rerun()
    else:
        time.sleep(1.5)
        st.session_state["show_welcome"] = False
        st.experimental_rerun()

# --- ページ選択（タブ切替）部分 --- #
if st.session_state["is_admin"]:
    pages = ["チャット", "Insights"]
else:
    pages = ["チャット"]

page = st.sidebar.selectbox("ページ選択", pages)

# --- ページ振り分け処理 --- #
def run_chat_page():
    st.title("LRADサポートチャット")
    st.caption("※このチャットボットはFAQとAIをもとに応答します。")
    st.write("（ここにチャット処理を実装）")

def run_insights_page():
    if not st.session_state.get("is_admin", False):
        st.error("このページへのアクセス権がありません。")
        st.stop()
    st.title("📊 LRADサポートチャット インサイト分析")
    st.write("（ここにInsightsページのコードを実装）")

if page == "チャット":
    run_chat_page()
elif page == "Insights":
    run_insights_page()
