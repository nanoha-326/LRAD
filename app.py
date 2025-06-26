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

st.set_page_config(page_title="LRADチャット", layout="centered")

# --- 言語設定とフォント ---
lang = st.sidebar.selectbox("言語を選択 / Select Language", ["日本語", "English"], index=0)
sidebar_title = "⚙️ 設定" if lang == "日本語" else "⚙️ Settings"
font_size_label = "文字サイズを選択" if lang == "日本語" else "Select Font Size"
font_size_options = ["小", "中", "大"] if lang == "日本語" else ["Small", "Medium", "Large"]
st.sidebar.title(sidebar_title)
font_size = st.sidebar.selectbox(font_size_label, font_size_options, index=1)

font_map_jp = {"小": "14px", "中": "18px", "大": "24px"}
font_map_en = {"Small": "14px", "Medium": "18px", "Large": "24px"}
selected_font_size = font_map_jp[font_size] if lang == "日本語" else font_map_en[font_size]

st.markdown(f"""
    <style>
        div[data-testid="stVerticalBlock"] * {{ font-size: {selected_font_size}; }}
        section[data-testid="stSidebar"] * {{ font-size: {selected_font_size}; }}
    </style>
""", unsafe_allow_html=True)

# --- 多言語対応テキスト ---
WELCOME_MESSAGES = [
    "ようこそ！LRADチャットボットへ。",
    "あなたの疑問にお応えします。",
    "LRAD専用チャットボットです。",
] if lang == "日本語" else [
    "Welcome to the LRAD Chat Assistant.",
    "Your questions, our answers.",
]

LOGIN_TITLE = "ログイン" if lang == "日本語" else "Login"
LOGIN_PASSWORD_LABEL = "パスワードを入力" if lang == "日本語" else "Enter Password"
LOGIN_ERROR_MSG = "パスワードが間違っています" if lang == "日本語" else "Incorrect password"
WELCOME_CAPTION = (
    "※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。"
    if lang == "日本語" else
    "This chatbot responds based on FAQ and AI, but may not answer all questions accurately."
)
CHAT_INPUT_PLACEHOLDER = "質問をどうぞ..." if lang == "日本語" else "Ask your question..."

# --- 状態管理 ---
CORRECT_PASSWORD = "mypassword"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "welcome_displayed" not in st.session_state:
    st.session_state.welcome_displayed = False
if "welcome_message" not in st.session_state:
    st.session_state.welcome_message = ""

# --- ログイン処理 ---
def password_check():
    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.title(LOGIN_TITLE)
            password = st.text_input(LOGIN_PASSWORD_LABEL, type="password")
            if st.form_submit_button(LOGIN_TITLE):
                if password == CORRECT_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.welcome_message = random.choice(WELCOME_MESSAGES)
                    st.session_state.welcome_displayed = False
                    st.experimental_rerun()
                else:
                    st.error(LOGIN_ERROR_MSG)
        st.stop()

password_check()

# --- ウェルカム画面（初回のみ） ---
def show_welcome():
    st.markdown(f"""
        <style>
        .fullscreen {{
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background-color: white; display: flex; justify-content: center;
            align-items: center; font-size: 48px; font-weight: bold;
            z-index: 9999; animation: fadeout 2s forwards;
        }}
        @keyframes fadeout {{ 0% {{ opacity: 1; }} 100% {{ opacity: 0; display: none; }} }}
        </style>
        <div class="fullscreen">{st.session_state.welcome_message}</div>
        <script>setTimeout(() => {{ window.location.reload(); }}, 2000);</script>
    """, unsafe_allow_html=True)

if not st.session_state.welcome_displayed:
    show_welcome()
    st.session_state.welcome_displayed = True
    st.stop()

# --- OpenAI クライアント初期化 ---
try:
    client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)
except Exception as e:
    st.error("OpenAI APIキーの取得に失敗しました。st.secretsの設定を確認してください。")
    st.stop()

# --- タイトル・ロゴ表示 ---
def get_base64_image(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

image_base64 = get_base64_image("LRADimg.png")
title_text = "LRADサポートチャット" if lang == "日本語" else "LRAD Support Chat"
st.markdown(f"""
    <div style="display:flex; align-items:center;">
        <img src="data:image/png;base64,{image_base64}" width="80" style="margin-right:10px;">
        <h1 style="margin:0; font-size:32px;">{title_text}</h1>
    </div>
""", unsafe_allow_html=True)

st.caption(WELCOME_CAPTION)

# --- FAQ 読み込み ---
def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        res = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return res.data[0].embedding
    except:
        return np.zeros(1536)

@st.cache_data
def load_faq(path="faq_all.csv"):
    df = pd.read_csv(path)
    df["embedding"] = df["質問"].apply(lambda x: get_embedding(str(x)))
    return df

faq_df = load_faq()

@st.cache_data

def load_common_faq():
    fname = "faq_common_jp.csv" if lang == "日本語" else "faq_common_en.csv"
    try:
        df = pd.read_csv(fname)
        return df
    except:
        return pd.DataFrame()

common_faq_df = load_common_faq()

# --- よくある質問 ---
with st.expander("\U0001F4A1 よくある質問" if lang == "日本語" else "\U0001F4A1 FAQ"):
    label = "\U0001F50E キーワードで検索" if lang == "日本語" else "\U0001F50E Search keyword"
    no_match = "一致するFAQが見つかりませんでした。" if lang == "日本語" else "No matching FAQ found."
    search_keyword = st.text_input(label, "")
    if not common_faq_df.empty:
        if search_keyword:
            cols = ["質問", "回答"] if lang == "日本語" else ["question", "answer"]
            df_filtered = common_faq_df[common_faq_df[cols[0]].str.contains(search_keyword, case=False, na=False) |
                                        common_faq_df[cols[1]].str.contains(search_keyword, case=False, na=False)]
        else:
            df_filtered = common_faq_df.sample(n=min(3, len(common_faq_df)))

        if df_filtered.empty:
            st.info(no_match)
        else:
            for _, row in df_filtered.iterrows():
                q = row["質問"] if lang == "日本語" else row["question"]
                a = row["回答"] if lang == "日本語" else row["answer"]
                st.markdown(f"**Q. {q}**")
                st.markdown(f"A. {a}")
                st.markdown("---")

# --- チャット処理 ---
def is_valid_input(text):
    text = text.strip()
    if not (3 <= len(text) <= 300): return False
    if len(re.findall(r"[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]", text)) / len(text) > 0.3: return False
    return True

def find_top_similar(q, df, k=1):
    q_vec = get_embedding(q)
    try:
        faq_vecs = np.stack(df["embedding"].to_numpy())
        sims = cosine_similarity([q_vec], faq_vecs)[0]
        idx = sims.argsort()[::-1][:k][0]
        return df.iloc[idx]["質問"], df.iloc[idx]["回答"]
    except:
        return None, None

def generate_response(user_q, ref_q, ref_a):
    prompt = f"あなたはLRADの専門家です。FAQ質問: {ref_q}\nFAQ回答: {ref_a}\nこの情報をもとに簡潔にユーザーの質問に答えてください。"
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_q}]
    try:
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return res.choices[0].message.content.strip()
    except:
        return "申し訳ありません、AIによる回答生成に失敗しました。"

def append_to_csv(q, a, path="chat_logs.csv"):
    try:
        df = pd.DataFrame([{"timestamp": pd.Timestamp.now().isoformat(), "question": q, "answer": a}])
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    except:
        pass

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

for q, a in st.session_state.chat_log:
    st.chat_message("user").write(q)
    if a: st.chat_message("assistant").write(a)

user_q = st.chat_input(CHAT_INPUT_PLACEHOLDER)
if user_q:
    if not is_valid_input(user_q):
        st.warning("入力が不正です。3〜300文字、記号率30%未満にしてください。")
    else:
        st.session_state.chat_log.append((user_q, None))
        st.experimental_rerun()

if st.session_state.chat_log and st.session_state.chat_log[-1][1] is None:
    last_q = st.session_state.chat_log[-1][0]
    ref_q, ref_a = find_top_similar(last_q, faq_df)
    answer = generate_response(last_q, ref_q, ref_a) if ref_q else "申し訳ありません、関連FAQが見つかりませんでした。"
    st.session_state.chat_log[-1] = (last_q, answer)
    append_to_csv(last_q, answer)
    st.experimental_rerun()
