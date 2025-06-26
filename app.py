# app.py

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

# 言語設定
lang = st.sidebar.selectbox("言語 / Language", ["日本語", "English"], index=0)
font_size_label = "文字サイズ" if lang == "日本語" else "Font Size"
font_size = st.sidebar.selectbox(font_size_label, ["小", "中", "大"] if lang == "日本語" else ["Small", "Medium", "Large"], index=1)

font_map_jp = {"小": "14px", "中": "18px", "大": "24px"}
font_map_en = {"Small": "14px", "Medium": "18px", "Large": "24px"}
font_css = font_map_jp[font_size] if lang == "日本語" else font_map_en[font_size]

st.markdown(f"""
<style>
div[data-testid="stVerticalBlock"] * {{ font-size: {font_css}; }}
section[data-testid="stSidebar"] * {{ font-size: {font_css}; }}
</style>
""", unsafe_allow_html=True)

# セッション状態
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = False
if "welcome_message" not in st.session_state:
    st.session_state.welcome_message = ""
if "fade_out" not in st.session_state:
    st.session_state.fade_out = False
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

WELCOME_MESSAGES_JP = ["ようこそ。LRADチャットボットへ", "質問があればお忘れなく"]
WELCOME_MESSAGES_EN = ["Welcome to LRAD Chatbot", "Ask anything about LRAD"]
WELCOME_MESSAGES = WELCOME_MESSAGES_JP if lang == "日本語" else WELCOME_MESSAGES_EN

# パスワード確認
def password_check():
    CORRECT_PASSWORD = "mypassword"
    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.title("Login" if lang != "日本語" else "ログイン")
            password = st.text_input("Password" if lang != "日本語" else "パスワード", type="password")
            if st.form_submit_button("Login"):
                if password == CORRECT_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.show_welcome = True
                    st.session_state.welcome_message = random.choice(WELCOME_MESSAGES)
                    st.session_state.fade_out = False
                    st.experimental_rerun()
                else:
                    st.error("Incorrect password" if lang != "日本語" else "パスワードが違います")
        st.stop()

password_check()

# ウェルカム表示
def show_welcome():
    st.markdown(f"""
    <style>
    .fullscreen {{
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background-color: white; display: flex;
        justify-content: center; align-items: center;
        font-size: 48px; font-weight: bold; z-index: 9999;
        animation: fadein 1s, fadeout 1s 2s forwards;
    }}
    @keyframes fadein {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes fadeout {{ from {{ opacity: 1; }} to {{ opacity: 0; }} }}
    </style>
    <div class="fullscreen">{st.session_state.welcome_message}</div>
    """, unsafe_allow_html=True)

if st.session_state.show_welcome:
    show_welcome()
    time.sleep(3)
    st.session_state.show_welcome = False
    st.experimental_rerun()

# OpenAI client
try:
    client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)
except Exception as e:
    st.error("OpenAI API key error.")
    st.stop()

# FAQ 読み込み
@st.cache_data
def load_faq(path="faq_all.csv"):
    df = pd.read_csv(path)
    df["embedding"] = df["質問"].apply(lambda x: get_embedding(str(x)))
    return df

@st.cache_data
def load_common_faq():
    if lang == "日本語":
        return pd.read_csv("faq_common_jp.csv")
    else:
        return pd.read_csv("faq_common_en.csv")

faq_df = load_faq()
common_faq_df = load_common_faq()

# タイトル表示
img_path = "LRADimg.png"
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
img_base64 = get_base64_image(img_path)
title = "LRADサポートチャット" if lang == "日本語" else "LRAD Support Chat"
st.markdown(f"""
<div style="display:flex;align-items:center;">
    <img src="data:image/png;base64,{img_base64}" width="80" style="margin-right:10px;">
    <h1 style="margin:0;font-size:32px;">{title}</h1>
</div>
""", unsafe_allow_html=True)

# よくある質問
with st.expander("💡 よくある質問" if lang == "日本語" else "💡 FAQ"):
    keyword = st.text_input("🔍 キーワード検索" if lang == "日本語" else "🔍 Search")
    df = common_faq_df
    if keyword:
        col_q = "質問" if lang == "日本語" else "question"
        col_a = "回答" if lang == "日本語" else "answer"
        df = df[df[col_q].str.contains(keyword, na=False) | df[col_a].str.contains(keyword, na=False)]
    for _, row in df.iterrows():
        st.markdown(f"**Q. {row[0]}**")
        st.markdown(f"A. {row[1]}")
        st.markdown("---")

# 基本処理
CHAT_PLACEHOLDER = "質問をどうぞ..." if lang == "日本語" else "Ask your question..."


def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        res = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return res.data[0].embedding
    except:
        return np.zeros(1536)


def find_similar(q, df):
    q_vec = get_embedding(q)
    matrix = np.stack(df.embedding.values)
    sims = cosine_similarity([q_vec], matrix)[0]
    idx = sims.argmax()
    return df.iloc[idx]


def generate_response(q, ref):
    system = "You are LRAD expert. Answer user in 200 chars using below FAQ." if lang != "日本語" else "あなたはLRADの専門家です。以下のFAQを参考に200文字以内で簡潔に回答してください。"
    messages = [
        {"role": "system", "content": f"{system}\nFAQ: {ref['質問']}\nA: {ref['回答']}"},
        {"role": "user", "content": q}
    ]
    try:
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return res.choices[0].message.content.strip()
    except:
        return "AIが回答を生成できませんでした"


user_q = st.chat_input(CHAT_PLACEHOLDER)

if user_q:
    st.session_state.chat_log.append((user_q, None))
    st.experimental_rerun()

if st.session_state.chat_log and st.session_state.chat_log[-1][1] is None:
    last_q = st.session_state.chat_log[-1][0]
    ref = find_similar(last_q, faq_df)
    answer = generate_response(last_q, ref)
    st.session_state.chat_log[-1] = (last_q, answer)
    st.experimental_rerun()

for q, a in st.session_state.chat_log:
    st.chat_message("user").write(q)
    if a:
        st.chat_message("assistant").write(a)
