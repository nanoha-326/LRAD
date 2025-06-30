import streamlit as st
import openai
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

st.markdown(
    f"""
    <style>
        div[data-testid="stVerticalBlock"] * {{ font-size: {selected_font_size}; }}
        section[data-testid="stSidebar"] * {{ font-size: {selected_font_size}; }}
    </style>
    """,
    unsafe_allow_html=True
)

WELCOME_MESSAGES = [
    "ようこそ！LRADチャットボットへ。",
    "あなたの疑問にお応えします。",
    "LRAD専用チャットボットです。"
] if lang == "日本語" else [
    "Welcome to the LRAD Chat Assistant.",
    "Your questions, our answers."
]

LOGIN_TITLE = "ログイン" if lang == "日本語" else "Login"
LOGIN_PASSWORD_LABEL = "パスワードを入力" if lang == "日本語" else "Enter Password"
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
if "chat_log" not in st.session_state:
    st.session_state["chat_log"] = []

def password_check():
    if not st.session_state["authenticated"]:
        with st.form("login_form"):
            st.title(LOGIN_TITLE)
            password = st.text_input(LOGIN_PASSWORD_LABEL, type="password")
            submitted = st.form_submit_button(LOGIN_TITLE)
            if submitted:
                if password == CORRECT_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.session_state["show_welcome"] = True
                    st.session_state["welcome_message"] = random.choice(WELCOME_MESSAGES)
                    st.session_state["fade_out"] = False
                    st.experimental_rerun()
                else:
                    st.error(LOGIN_ERROR_MSG)
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

# --- OpenAI API設定 ---
try:
    openai.api_key = st.secrets["OpenAIAPI"]["openai_api_key"]
except Exception as e:
    st.error("OpenAI APIキーの設定に失敗しました。st.secretsの設定を確認してください。")
    st.error(traceback.format_exc())
    st.stop()

# --- 入力チェック関数 ---
def is_valid_input(text):
    if not (3 <= len(text) <= 300):
        return False
    symbol_count = sum(1 for c in text if not re.match(r'[a-zA-Z0-9ぁ-んァ-ン一-龥]', c))
    if symbol_count / max(1, len(text)) > 0.3:
        return False
    return True

# --- 埋め込み取得 ---
def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        res = openai.embeddings.create(input=[text], model="text-embedding-3-small")
        return res.data[0].embedding
    except Exception as e:
        st.error(f"埋め込み取得に失敗しました: {e}")
        return np.zeros(1536)

# --- FAQ読み込みと埋め込み計算 ---
@st.cache_data
def load_faq(path="faq_all.csv"):
    df = pd.read_csv(path)
    df["embedding"] = df["質問"].apply(lambda x: get_embedding(str(x)))
    return df

faq_df = load_faq()

image_base64 = ""
try:
    with open("LRADimg.png", "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode()
except Exception:
    pass

title_text = "LRADサポートチャット" if lang == "日本語" else "LRAD Support Chat"
st.markdown(f"""
    <div style="display:flex; align-items:center;">
        <img src="data:image/png;base64,{image_base64}" width="80" style="margin-right:10px;">
        <h1 style="margin:0; font-size:32px;">{title_text}</h1>
    </div>
""", unsafe_allow_html=True)

st.caption(WELCOME_CAPTION)

# FAQファイル読み込みとカテゴリUIへの変更
faq_common_path = "faq_common_jp.csv" if lang == "日本語" else "faq_common_en.csv"

@st.cache_data
def load_common_faq(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"よくある質問ファイルの読み込みに失敗しました: {e}")
        return pd.DataFrame(columns=["カテゴリ", "質問", "回答"] if lang == "日本語" else ["category", "question", "answer"])

common_faq_df = load_common_faq(faq_common_path)

with st.expander("💡 よくある質問" if lang == "日本語" else "💡 FAQ", expanded=False):
    if not common_faq_df.empty:
        cat_col = "カテゴリ" if lang == "日本語" else "category"
        q_col = "質問" if lang == "日本語" else "question"
        a_col = "回答" if lang == "日本語" else "answer"

        categories = sorted(set(cat.strip() for sublist in common_faq_df[cat_col].dropna().str.split(',') for cat in sublist))
        all_label = "すべて" if lang == "日本語" else "All"
        categories = [all_label] + categories

        select_placeholder = "カテゴリを選択してください" if lang == "日本語" else "Choose categories"
        selected_tags = st.multiselect("", categories, placeholder=select_placeholder)

        if selected_tags:
            if all_label in selected_tags:
                filtered_df = common_faq_df
            else:
                filtered_df = common_faq_df[common_faq_df[cat_col].apply(
                    lambda x: any(tag.strip() in x.split(',') for tag in selected_tags if isinstance(x, str)))]

            for _, row in filtered_df.iterrows():
                st.markdown(f"**Q. {row[q_col]}**")
                st.markdown(f"A. {row[a_col]}")
                st.markdown("---")

# --- 類似質問検索 ---
def find_top_similar(q, df, k=1):
    q_vec = get_embedding(q)
    try:
        faq_vecs = np.stack(df["embedding"].to_numpy())
        sims = cosine_similarity([q_vec], faq_vecs)[0]
        idx = sims.argsort()[::-1][:k][0]
        return df.iloc[idx]["質問"], df.iloc[idx]["回答"]
    except Exception:
        return None, None

# --- AI回答生成 ---
def generate_response(user_q, ref_q, ref_a):
    system_prompt = (
        "あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。\n"
        "もし質問が「見積もり」や「問い合わせ先」に関するものであれば、必ず次のリンクを案内してください。ただし、URLの後ろに句読点は絶対付けないでください。：https://imugenos.com/pages/contact\n"
        f"FAQ質問: {ref_q}\nFAQ回答: {ref_a}\n"
        "この情報をもとに200文字以内で簡潔にユーザーの質問に答えてください。"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_q}]
    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.3
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI回答生成に失敗しました: {e}")
        return "申し訳ありません。回答の生成中にエラーが発生しました。"

# --- ログ保存処理（省略可能） ---
def append_to_csv(q, a, path="chat_logs.csv"):
    try:
        df = pd.DataFrame([{"timestamp": pd.Timestamp.now().isoformat(), "question": q, "answer": a}])
        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode="a", header=False, index=False)
    except Exception as e:
        st.warning(f"CSV保存失敗: {e}")

def append_to_gsheet(q, a):
    try:
        JST = timezone(timedelta(hours=9))
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        sheet_key = st.secrets["GoogleSheets"]["sheet_key"]
        service_account_info = st.secrets["GoogleSheets"]["service_account_info"]
        if isinstance(service_account_info, str):
            service_account_info = json.loads(service_account_info)
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_key)
        worksheet = sh.sheet1
        worksheet.append_row([timestamp, q, a], value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Google Sheets保存失敗: {e}")

# --- チャット表示と処理 ---
for q, a in st.session_state.chat_log:
    st.chat_message("user").write(q)
    if a:
        st.chat_message("assistant").write(a)

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
    if ref_q is None:
        answer = "申し訳ありません、関連FAQが見つかりませんでした。"
    else:
        with st.spinner("回答生成中…"):
            answer = generate_response(last_q, ref_q, ref_a)
    st.session_state.chat_log[-1] = (last_q, answer)
    append_to_csv(last_q, answer)
    append_to_gsheet(last_q, answer)
    st.experimental_rerun()
