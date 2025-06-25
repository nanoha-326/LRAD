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

# --- 通常のログイン＆ウェルカムメッセージ部分 --- #

CORRECT_PASSWORD = "mypassword"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
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

# --- OpenAI初期化 --- #
try:
    client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)
except Exception as e:
    st.error("OpenAI APIキーの取得に失敗しました。st.secretsの設定を確認してください。")
    st.error(traceback.format_exc())
    st.stop()

# --- 画像表示 --- #
def get_base64_image(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.warning(f"画像の読み込みに失敗しました: {e}")
        return ""

image_base64 = get_base64_image("LRADimg.png")

st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
        <img src="data:image/png;base64,{image_base64}" width="80" style="margin-right:10px;">
        <h1 style="margin:0;">LRADサポートチャット</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

# --- 入力バリデーション --- #
def is_valid_input(text: str) -> bool:
    text = text.strip()
    if not (3 <= len(text) <= 300):
        return False
    if len(re.findall(r'[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]', text)) / len(text) > 0.3:
        return False
    try:
        unicodedata.normalize("NFKC", text).encode("utf-8")
    except UnicodeError:
        return False
    return True

# --- Embedding取得 --- #
def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        res = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return res.data[0].embedding
    except Exception as e:
        st.error(f"埋め込み取得に失敗しました: {e}")
        return np.zeros(1536)

# --- FAQ読み込み --- #
@st.cache_data
def load_faq(path="faq_all_with_embed.csv"):
    def parse_embedding(val):
        try:
            return np.array(json.loads(val))
        except Exception:
            return np.zeros(1536)
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(parse_embedding)
    return df

faq_df = load_faq()

@st.cache_data
def load_common_faq(path="faq_common.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"よくある質問ファイルの読み込みに失敗しました: {e}")
        return pd.DataFrame(columns=["質問", "回答"])

common_faq_df = load_common_faq()

with st.expander("💡 よくある質問", expanded=False):
    if not common_faq_df.empty:
        sample = common_faq_df.sample(1)
        for _, row in sample.iterrows():
            st.markdown(f"**Q. {row['質問']}**\n\nA. {row['回答']}")

# --- FAQ類似質問検索 --- #
def find_top_similar(q, df, k=1):
    q_vec = get_embedding(q)
    try:
        faq_vecs = np.stack(df["embedding"].to_numpy())
        sims = cosine_similarity([q_vec], faq_vecs)[0]
        idx = sims.argsort()[::-1][:k][0]
        return df.iloc[idx]["質問"], df.iloc[idx]["回答"]
    except Exception:
        return None, None

# --- AI回答生成 --- #
def generate_response(user_q, ref_q, ref_a):
    system_prompt = (
        "あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。\n"
        f"FAQ質問: {ref_q}\nFAQ回答: {ref_a}\n"
        "この情報をもとに200文字以内で簡潔にユーザーの質問に答えてください。"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_q}]
    try:
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=0.3)
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI回答生成に失敗しました: {e}")
        return "申し訳ありません、AIによる回答生成に失敗しました。"

# --- チャットログ保存 --- #
def append_to_csv(q, a, path="chat_logs.csv"):
    try:
        df = pd.DataFrame([{ "timestamp": pd.Timestamp.now().isoformat(), "question": q, "answer": a }])
        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode='a', header=False, index=False)
    except Exception as e:
        st.warning(f"CSVへの保存に失敗しました: {e}")

def append_to_gsheet(q, a):
    try:
        JST = timezone(timedelta(hours=9))
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        sheet_key = st.secrets["GoogleSheets"]["sheet_key"]
        service_account_info = st.secrets["GoogleSheets"]["service_account_info"]
        if isinstance(service_account_info, str):
            service_account_info = json.loads(service_account_info)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_key)
        worksheet = sh.sheet1
        worksheet.append_row([timestamp, q, a], value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Google Sheetsへの保存に失敗しました: {e}")

# --- セッション初期化 --- #
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# --- 過去ログ表示 --- #
for q, a in st.session_state.chat_log:
    st.chat_message("user").write(q)
    if a:
        st.chat_message("assistant").write(a)

# --- ユーザー入力 --- #
user_q = st.chat_input("質問をどうぞ...")

if user_q:
    if not is_valid_input(user_q):
        st.warning("入力が不正です。3〜300文字、記号率30%未満にしてください。")
    else:
        ref_q, ref_a = find_top_similar(user_q, faq_df)
        if ref_q is None:
            answer = "申し訳ありません、関連FAQが見つかりませんでした。"
        else:
            with st.spinner("回答を生成中..."):
                answer = generate_response(user_q, ref_q, ref_a)
        st.session_state.chat_log.append((user_q, answer))
        append_to_csv(user_q, answer)
        append_to_gsheet(user_q, answer)
        st.experimental_rerun()
