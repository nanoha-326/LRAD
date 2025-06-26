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

# ------------------------ 言語・フォント設定 ------------------------
lang = st.sidebar.selectbox("言語を選択 / Select Language", ["日本語", "English"], index=0)
font_size_label = "文字サイズを選択" if lang == "日本語" else "Select Font Size"
font_size_options = ["小", "中", "大"] if lang == "日本語" else ["Small", "Medium", "Large"]
font_size_map = {"小": "14px", "中": "18px", "大": "24px", "Small": "14px", "Medium": "18px", "Large": "24px"}
font_size = st.sidebar.selectbox(font_size_label, font_size_options, index=1)
st.markdown(f"""
    <style>
        div[data-testid="stVerticalBlock"] * {{ font-size: {font_size_map[font_size]}; }}
        section[data-testid="stSidebar"] * {{ font-size: {font_size_map[font_size]}; }}
    </style>
""", unsafe_allow_html=True)

# ------------------------ セッション初期化 ------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "show_welcome" not in st.session_state:
    st.session_state["show_welcome"] = False
if "fade_out" not in st.session_state:
    st.session_state["fade_out"] = False
if "welcome_message" not in st.session_state:
    st.session_state["welcome_message"] = ""
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

WELCOME_MESSAGES = [
    "ようこそ！LRADチャットボットへ。",
    "あなたの疑問にお応えします。",
    "LRAD専用チャットボットです。",
] if lang == "日本語" else [
    "Welcome to the LRAD Chat Assistant.",
    "Your questions, our answers."
]

# ------------------------ 認証 ------------------------
def password_check():
    if not st.session_state["authenticated"]:
        with st.form("login_form"):
            st.title("ログイン" if lang == "日本語" else "Login")
            password = st.text_input("パスワードを入力" if lang == "日本語" else "Enter Password", type="password")
            submitted = st.form_submit_button("ログイン" if lang == "日本語" else "Login")
            if submitted:
                if password == "mypassword":
                    st.session_state["authenticated"] = True
                    st.session_state["show_welcome"] = True
                    st.session_state["fade_out"] = False
                    st.session_state["welcome_message"] = random.choice(WELCOME_MESSAGES)
                    st.experimental_rerun()
                else:
                    st.error("パスワードが間違っています" if lang == "日本語" else "Incorrect password")
        st.stop()

password_check()

# ------------------------ Welcome画面 ------------------------
def show_welcome_screen():
    st.markdown(f"""
    <style>
    .fullscreen {{
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background-color: white; display: flex;
        justify-content: center; align-items: center;
        font-size: 60px; font-weight: bold;
        animation: fadein 1.2s forwards; z-index: 9999;
    }}
    .fadeout {{ animation: fadeout 1.5s forwards; }}
    @keyframes fadein {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes fadeout {{ from {{ opacity: 1; }} to {{ opacity: 0; }} }}
    </style>
    <div class="fullscreen {'fadeout' if st.session_state['fade_out'] else ''}">
        {st.session_state['welcome_message']}
    </div>
    """, unsafe_allow_html=True)

if st.session_state["show_welcome"]:
    show_welcome_screen()
    if not st.session_state["fade_out"]:
        time.sleep(2)
        st.session_state["fade_out"] = True
        st.experimental_rerun()
    else:
        time.sleep(1.5)
        st.session_state["show_welcome"] = False
        st.experimental_rerun()

# ------------------------ タイトル ------------------------
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

image_base64 = get_base64_image("LRADimg.png")

st.markdown(f"""
<div style="display:flex; align-items:center;">
    <img src="data:image/png;base64,{image_base64}" width="80" style="margin-right:10px;">
    <h1 style="margin:0; font-size:32px;">
        {'LRADサポートチャット' if lang == '日本語' else 'LRAD Support Chat'}
    </h1>
</div>
""", unsafe_allow_html=True)

st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。" if lang == "日本語" else "This chatbot responds based on FAQ and AI, but may not answer all questions accurately.")

# ------------------------ OpenAIクライアント ------------------------
try:
    client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)
except Exception:
    st.error("OpenAI APIキーの取得に失敗しました")
    st.stop()

# ------------------------ FAQ読み込み ------------------------
@st.cache_data
def load_faq(path="faq_all.csv"):
    df = pd.read_csv(path)
    df["embedding"] = df["質問"].apply(lambda x: get_embedding(str(x)))
    return df

def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        res = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return res.data[0].embedding
    except:
        return np.zeros(1536)

faq_df = load_faq()

# ------------------------ 類似検索＆応答 ------------------------
def find_top_similar(q, df, k=1):
    q_vec = get_embedding(q)
    faq_vecs = np.stack(df["embedding"].to_numpy())
    sims = cosine_similarity([q_vec], faq_vecs)[0]
    idx = sims.argsort()[::-1][:k][0]
    return df.iloc[idx]["質問"], df.iloc[idx]["回答"]

def generate_response(user_q, ref_q, ref_a):
    sys_prompt = f"""あなたはLRADの専門家です。\n
FAQ質問: {ref_q}\nFAQ回答: {ref_a}\n
この情報をもとに200文字以内でユーザーの質問に答えてください。"""
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_q}]
    res = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=0.3)
    return res.choices[0].message.content.strip()

# ------------------------ チャットUI ------------------------
for q, a in st.session_state.chat_log:
    st.chat_message("user").write(q)
    if a:
        st.chat_message("assistant").write(a)

user_q = st.chat_input("質問をどうぞ..." if lang == "日本語" else "Ask your question...")

if user_q:
    st.session_state.chat_log.append((user_q, None))
    st.experimental_rerun()

if st.session_state.chat_log and st.session_state.chat_log[-1][1] is None:
    last_q = st.session_state.chat_log[-1][0]
    ref_q, ref_a = find_top_similar(last_q, faq_df)
    if ref_q:
        with st.spinner("回答生成中..."):
            ans = generate_response(last_q, ref_q, ref_a)
    else:
        ans = "申し訳ありません、関連FAQが見つかりませんでした。"
    st.session_state.chat_log[-1] = (last_q, ans)
    st.experimental_rerun()
