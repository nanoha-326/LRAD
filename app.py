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

# --- 言語設定とサイドバーUI ---
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

# --- 定数・メッセージ ---
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

# --- セッションステート初期化 ---
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

# --- パスワード認証フォーム ---
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

# --- ウェルカムメッセージ表示 ---
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

# --- OpenAIクライアント初期化 ---
try:
    client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)
except Exception as e:
    st.error("OpenAI APIキーの取得に失敗しました。st.secretsの設定を確認してください。")
    st.error(traceback.format_exc())
    st.stop()

# --- テキスト埋め込み取得関数 ---
def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        res = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return res.data[0].embedding
    except Exception as e:
        st.error(f"埋め込み取得に失敗しました: {e}")
        return np.zeros(1536)

# --- FAQデータ読み込み（チャット用） ---
@st.cache_data
def load_faq(path="faq_all.csv"):
    df = pd.read_csv(path)
    # FAQの「質問」列を使い埋め込み計算（キャッシュされる）
    df["embedding"] = df["質問"].apply(lambda x: get_embedding(str(x)))
    return df

faq_df = load_faq()

# --- FAQデータ読み込み（よくある質問用） ---
faq_common_path = "faq_common_jp.csv" if lang == "日本語" else "faq_common_en.csv"

@st.cache_data
def load_common_faq(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"よくある質問ファイルの読み込みに失敗しました: {e}")
        # 日本語・英語でカラムを分ける
        cols = ["質問", "回答"] if lang == "日本語" else ["question", "answer"]
        return pd.DataFrame(columns=cols)

common_faq_df = load_common_faq(faq_common_path)

# --- 画面タイトル＆ロゴ表示 ---
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

# --- よくある質問表示エリア ---
with st.expander("💡 よくある質問" if lang == "日本語" else "💡 FAQ", expanded=False):
    if not common_faq_df.empty:
        search_label = "🔎 キーワードで検索" if lang == "日本語" else "🔎 Search keyword"
        no_match_msg = "一致するFAQが見つかりませんでした。" if lang == "日本語" else "No matching FAQ found."
        search_keyword = st.text_input(search_label, "")
        col_q = "質問" if lang == "日本語" else "question"
        col_a = "回答" if lang == "日本語" else "answer"

        if search_keyword:
            df_filtered = common_faq_df[
                common_faq_df[col_q].str.contains(search_keyword, na=False, case=False) |
                common_faq_df[col_a].str.contains(search_keyword, na=False, case=False)
            ]
            if df_filtered.empty:
                st.info(no_match_msg)
            else:
                for _, row in df_filtered.iterrows():
                    st.markdown(f"**Q. {row[col_q]}**")
                    st.markdown(f"A. {row[col_a]}")
                    st.markdown("---")
        else:
            sample = common_faq_df.sample(n=min(3, len(common_faq_df)))
            for _, row in sample.iterrows():
                st.markdown(f"**Q. {row[col_q]}**")
                st.markdown(f"A. {row[col_a]}")
                st.markdown("---")

# --- 入力検証関数 ---
def is_valid_input(text: str) -> bool:
    text = text.strip()
    if not (3 <= len(text) <= 300):
        return False
    if len(re.findall(r"[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]", text)) / max(len(text),1) > 0.3:
        return False
    try:
        unicodedata.normalize("NFKC", text).encode("utf-8")
    except UnicodeError:
        return False
    return True

# --- 類似FAQ検索関数 ---
def find_top_similar(q, df, k=1):
    q_vec = get_embedding(q)
    try:
        faq_vecs = np.stack(df["embedding"].to_numpy())
        sims = cosine_similarity([q_vec], faq_vecs)[0]
        idx = sims.argsort()[::-1][:k][0]
        return df.iloc[idx]["質問"], df.iloc[idx]["回答"]
    except Exception:
        return None, None

# --- 回答生成関数 ---
def generate_response(user_q, ref_q, ref_a):
    system_prompt = (
        "あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。\n"
        f"FAQ質問: {ref_q}\nFAQ回答: {ref_a}\n"
        "この情報をもとに200文字以内で簡潔にユーザーの質問に答えてください。"
        if lang == "日本語" else
        "You are an expert on LRAD (far-infrared electronic pyrolysis device).\n"
        f"FAQ Question: {ref_q}\nFAQ Answer: {ref_a}\n"
        "Answer the user's question concisely within 200 characters based on this information."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_q}]
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.3
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI回答生成に失敗しました: {e}")
        return "申し訳ありません、AIによる回答生成に失敗しました。" if lang == "日本語" else "Sorry, failed to generate AI response."

# --- CSVログ保存関数 ---
def append_to_csv(q, a, path="chat_logs.csv"):
    try:
        df = pd.DataFrame([{"timestamp": pd.Timestamp.now().isoformat(), "question": q, "answer": a}])
        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode="a", header=False, index=False)
    except Exception as e:
        st.warning(f"CSVへの保存に失敗しました: {e}")

# --- Google Sheetsログ保存関数 ---
def append_to_gsheet(q, a):
    try:
        JST = timezone(timedelta(hours=9))
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        sheet_key = st.secrets.gsheet.key
        credentials_dict = st.secrets.gsheet.credentials
        creds = Credentials.from_service_account_info(credentials_dict)
        gc = gspread.authorize(creds)
        ws = gc.open_by_key(sheet_key).sheet1
        ws.append_row([timestamp, q, a])
    except Exception as e:
        st.warning(f"Google Sheetsへの保存に失敗しました: {e}")

# --- チャットUI ---
st.write("---")
st.markdown(f"### {'質問入力' if lang == '日本語' else 'Ask a Question'}")

user_input = st.text_input("", placeholder=CHAT_INPUT_PLACEHOLDER, key="input")

if user_input and is_valid_input(user_input):
    # 類似FAQ検索（上位1件）
    ref_q, ref_a = find_top_similar(user_input, faq_df, k=1)
    if ref_q is None:
        response = "申し訳ありません、関連するFAQが見つかりませんでした。" if lang == "日本語" else "Sorry, no related FAQ found."
    else:
        response = generate_response(user_input, ref_q, ref_a)

    # チャットログ更新
    st.session_state.chat_log.append({"user": user_input, "bot": response})

    # ログ保存
    append_to_csv(user_input, response)
    append_to_gsheet(user_input, response)

# チャットログ表示（下から上へ）
if st.session_state.chat_log:
    for chat in reversed(st.session_state.chat_log):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
