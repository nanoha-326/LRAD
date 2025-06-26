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

# 言語設定とフォントサイズ
lang = st.sidebar.selectbox("言語を選択 / Select Language", ["日本語", "English"], index=0)
font_size = st.sidebar.selectbox(
    "文字サイズを選択" if lang == "日本語" else "Select Font Size",
    ["小", "中", "大"] if lang == "日本語" else ["Small", "Medium", "Large"],
    index=1,
)
font_size_map = {"小": "14px", "中": "18px", "大": "24px", "Small": "14px", "Medium": "18px", "Large": "24px"}
selected_font_size = font_size_map[font_size]

st.markdown(f"""
<style>
div[data-testid="stVerticalBlock"] * {{ font-size: {selected_font_size}; }}
section[data-testid="stSidebar"] * {{ font-size: {selected_font_size}; }}
</style>
""", unsafe_allow_html=True)

# Welcome メッセージ
WELCOME_MESSAGES = {
    "日本語": ["ようこそ！LRADチャットボットへ。", "あなたの疑問にお応えします。", "LRAD専用チャットボットです。"],
    "English": ["Welcome to the LRAD Chat Assistant.", "Your questions, our answers."]
}

st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("show_welcome", False)
st.session_state.setdefault("welcome_message", "")

# パスワード認証
if not st.session_state["authenticated"]:
    with st.form("login_form"):
        st.title("ログイン" if lang == "日本語" else "Login")
        password = st.text_input("パスワードを入力" if lang == "日本語" else "Enter Password", type="password")
        if st.form_submit_button("ログイン" if lang == "日本語" else "Login"):
            if password == "mypassword":
                st.session_state["authenticated"] = True
                st.session_state["show_welcome"] = True
                st.session_state["welcome_message"] = random.choice(WELCOME_MESSAGES[lang])
                st.experimental_rerun()
            else:
                st.error("パスワードが間違っています" if lang == "日本語" else "Incorrect password")
    st.stop()

# Welcome表示
if st.session_state["show_welcome"]:
    st.markdown(f"""
    <style>
    .fullscreen {{
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background-color: white; display: flex; justify-content: center;
        align-items: center; font-size: 48px; font-weight: bold; z-index: 9999;
    }}
    </style>
    <div class="fullscreen">{st.session_state['welcome_message']}</div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    st.session_state["show_welcome"] = False
    st.experimental_rerun()

# OpenAI 接続
try:
    client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)
except Exception as e:
    st.error("OpenAI APIキー取得に失敗: st.secretsを確認してください")
    st.error(traceback.format_exc())
    st.stop()

# タイトル画像
def get_base64_image(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

image_base64 = get_base64_image("LRADimg.png")
title_text = "LRADサポートチャット" if lang == "日本語" else "LRAD Support Chat"
st.markdown(f"""
<div style="display:flex; align-items:center;">
  <img src="data:image/png;base64,{image_base64}" width="80" style="margin-right:10px;">
  <h1 style="margin:0; font-size:32px;">{title_text}</h1>
</div>
""", unsafe_allow_html=True)

st.caption("※このチャットボットはFAQとAIをもとに応答しますが、正確な回答を保証するものではありません。" if lang == "日本語" else "This chatbot uses FAQ and AI to respond but may not be 100% accurate.")

# FAQデータ読み込み
@st.cache_data
def load_common_faq():
    path = "faq_common_jp.csv" if lang == "日本語" else "faq_common_en.csv"
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame(columns=["質問", "回答"] if lang == "日本語" else ["question", "answer"])

common_faq_df = load_common_faq()

# FAQ表示 + キーワード検索
with st.expander("💡 よくある質問" if lang == "日本語" else "💡 FAQ"):
    search_label = "🔎 キーワードで検索" if lang == "日本語" else "🔎 Search keyword"
    search_keyword = st.text_input(search_label, "")
    
    if lang == "日本語":
        display_cols = ["質問", "回答"]
    else:
        display_cols = ["question", "answer"]

    if search_keyword:
        df_filtered = common_faq_df[
            common_faq_df[display_cols[0]].str.contains(search_keyword, case=False, na=False) |
            common_faq_df[display_cols[1]].str.contains(search_keyword, case=False, na=False)
        ]
    else:
        df_filtered = common_faq_df.sample(n=min(3, len(common_faq_df)))

    if df_filtered.empty:
        st.info("一致するFAQが見つかりませんでした。" if lang == "日本語" else "No matching FAQ found.")
    else:
        for _, row in df_filtered.iterrows():
            st.markdown(f"**Q. {row[display_cols[0]]}**")
            st.markdown(f"A. {row[display_cols[1]]}")
            st.markdown("---")

# 以下にチャットUIや回答生成などを続けて記述（省略）
