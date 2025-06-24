# LRADサポートチャット（改良版：エラー処理・UI改善・キャッシュ管理）
import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, random, re, unicodedata, json, base64
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ページ設定
st.set_page_config(page_title="LRADサポートチャット", layout="centered")

# セキュアに管理したい場合は secrets.toml などへ
CORRECT_PASSWORD = "mypassword"

def password_check():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        with st.form("login_form"):
            st.write("🔒 パスワードを入力してください")
            password = st.text_input("パスワード", type="password")
            submitted = st.form_submit_button("ログイン")

            if submitted:
                if password == CORRECT_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.experimental_rerun()  # ← 🔁 ここでアプリを再起動させる（重要）
                else:
                    st.error("パスワードが間違っています")
                    st.stop()

    if not st.session_state["authenticated"]:
        st.stop()

# ログインチェック
password_check()

# ここから下はログイン後のアプリ内容
st.title("チャットボットアプリ")
st.write("ようこそ！これはパスワードで保護されたアプリです。")

# OpenAIキー
try:
    client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)
except Exception as e:
    st.error("OpenAI APIキーの取得に失敗しました。st.secretsの設定を確認してください。")
    st.stop()

# Google Sheets保存（エラー処理追加）
def append_to_gsheet(question, answer):
    try:
        from datetime import datetime  # 念のため関数内でも読み込む
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sheet_key = st.secrets["GoogleSheets"]["sheet_key"]
        service_account_info = st.secrets["GoogleSheets"]["service_account_info"]

        if isinstance(service_account_info, str):
            service_account_info = json.loads(service_account_info)

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",  # ✅ 新しい Sheets API スコープ
            "https://www.googleapis.com/auth/drive"
        ]

        creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_key)
        worksheet = sh.sheet1

        worksheet.append_row([timestamp, question, answer], value_input_option="USER_ENTERED")
    except Exception as e:
        st.error(f"Google Sheets への保存に失敗しました: {repr(e)}")


# CSV保存（エラー処理追加）
def append_to_csv(question, answer, path="chat_logs.csv"):
    try:
        df = pd.DataFrame([{ "timestamp": pd.Timestamp.now().isoformat(), "question": question, "answer": answer }])
        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode='a', header=False, index=False)
    except Exception as e:
        st.warning(f"CSVへの保存に失敗しました: {e}")

# CSS注入
def inject_custom_css(selected_size):
    st.markdown(
        f"""
        <style>
        .chat-text, .stCaption, .css-ffhzg2 p, .stTextInput > label {{
            font-size: {selected_size} !important;
        }}
        .stTextInput > div > div > input {{
            font-size: {selected_size} !important;
        }}
        ::placeholder {{
            font-size: {selected_size} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Embedding取得（エラー処理追加）
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    try:
        res = client.embeddings.create(input=[text], model=model)
        return res.data[0].embedding
    except Exception as e:
        st.error(f"埋め込み取得に失敗しました: {e}")
        return np.zeros(1536)

# 入力チェック
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

# FAQ埋め込みキャッシュ再計算用ボタン
def recalc_faq_embeddings(path="faq_all.csv", cached="faq_all_with_embed.csv"):
    try:
        df = pd.read_csv(path)
        with st.spinner("全FAQへ埋め込み計算中…（再計算）"):
            df["embedding"] = df["質問"].apply(get_embedding)
        df["embedding"] = df["embedding"].apply(lambda x: json.dumps(x.tolist()) if hasattr(x, "tolist") else x)
        df.to_csv(cached, index=False)
        st.success("FAQ埋め込みの再計算が完了しました。")
    except Exception as e:
        st.error(f"FAQ埋め込み再計算に失敗しました: {e}")

def get_modified_time(path):
    try:
        return os.path.getmtime(path)
    except:
        return 0
        
# CSV読み込み
@st.cache_data(show_spinner=False)
def load_faq_all(path="faq_all.csv", cached="faq_all_with_embed.csv", mtime=None):
    def parse_embedding(val):
        if isinstance(val, str):
            try:
                return np.array(json.loads(val))
            except Exception:
                pass
        elif isinstance(val, list) or isinstance(val, np.ndarray):
            return np.array(val)
        return np.zeros(1536)

    if os.path.exists(cached):
        try:
            df = pd.read_csv(cached)
            df["embedding"] = df["embedding"].apply(parse_embedding)
        except Exception as e:
            st.warning(f"キャッシュ読み込みに失敗: {e}。再計算をお試しください。")
            df = pd.read_csv(path)
            with st.spinner("全FAQへ埋め込み計算中…（初回のみ）"):
                df["embedding"] = df["質問"].apply(get_embedding)
            df["embedding"] = df["embedding"].apply(lambda x: json.dumps(x.tolist()) if hasattr(x, "tolist") else x)
            df.to_csv(cached, index=False)
            df["embedding"] = df["embedding"].apply(parse_embedding)
    else:
        df = pd.read_csv(path)
        with st.spinner("全FAQへ埋め込み計算中…（初回のみ）"):
            df["embedding"] = df["質問"].apply(get_embedding)
        df["embedding"] = df["embedding"].apply(lambda x: json.dumps(x.tolist()) if hasattr(x, "tolist") else x)
        df.to_csv(cached, index=False)
        df["embedding"] = df["embedding"].apply(parse_embedding)
    return df

mtime = get_modified_time("faq_all.csv") + get_modified_time("faq_all_with_embed.csv")
df = load_faq_all(path="faq_all.csv", cached="faq_all_with_embed.csv", mtime=mtime)

@st.cache_data(show_spinner=False)
def load_faq_common(path="faq_common.csv"):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"よくある質問ファイルの読み込みに失敗しました: {e}")
        return pd.DataFrame(columns=["質問", "回答"])

# サイドバー設定
st.sidebar.title("⚙️ 表示設定")
font_size = st.sidebar.selectbox("文字サイズを選んでください", ["小", "中", "大"])
font_size_map = {"小": "14px", "中": "18px", "大": "24px"}
img_width_map = {"小": 60, "中": 80, "大": 110}
selected_font = font_size_map[font_size]
selected_img = img_width_map[font_size]

# チャット履歴保存件数
max_log = st.sidebar.slider("チャット履歴の保存件数", min_value=10, max_value=200, value=100, step=10)

# チャット履歴の表示順
log_order = st.sidebar.radio("チャット履歴の表示順", ["新しい順", "古い順"])

# FAQ埋め込みキャッシュ再計算ボタン
if st.sidebar.button("FAQ埋め込みキャッシュ再計算"):
    recalc_faq_embeddings()
    st.cache_data.clear()
    st.experimental_rerun()

# FAQキャッシュクリアボタン
if st.sidebar.button("FAQキャッシュクリア"):
    st.cache_data.clear()
    st.success("FAQキャッシュをクリアしました。")
    st.experimental_rerun()

inject_custom_css(selected_font)

# ヘッダー画像
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
    <div style="display:flex; align-items:center;" class="chat-header">
        <img src="data:image/png;base64,{image_base64}"
             width="{selected_img}px" style="margin-right:10px;">
        <h1 style="margin:0; font-size:40px; font-weight:bold;">LRADサポートチャット</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

# FAQ・共通FAQ読み込み
faq_df = load_faq_all()
common_faq_df = load_faq_common()

# よくある質問表示
def display_random_common_faqs(common_faq_df, n=3):
    if len(common_faq_df) == 0:
        st.info("よくある質問がありません。")
        return
    sampled = common_faq_df.sample(min(n, len(common_faq_df)))
    for i, row in enumerate(sampled.itertuples(), 1):
        question = getattr(row, "質問", "（質問が不明です）")
        answer = getattr(row, "回答", "（回答が不明です）")
        st.markdown(
            f'<div class="chat-text"><b>❓ {question}</b><br>🅰️ {answer}</div><hr>',
            unsafe_allow_html=True
        )

st.markdown("### 💡 よくある質問（ランダム表示）")
display_random_common_faqs(common_faq_df, n=3)
st.divider()

# 類似質問検索
def find_top_similar(q, df, k=1):
    if len(q.strip()) < 2:
        return None, None
    q_vec = get_embedding(q)
    try:
        faq_vecs = np.stack(df["embedding"].to_numpy())
        sims = cosine_similarity([q_vec], faq_vecs)[0]
        idx = sims.argsort()[::-1][:k][0]
        return df.iloc[idx]["質問"], df.iloc[idx]["回答"]
    except Exception as e:
        st.warning(f"類似質問検索に失敗しました: {e}")
        return None, None

# チャット履歴要約
def summarize_chat_log(log, max_turns=5):
    if not log:
        return ""
    recent = log[:max_turns]
    prompt = "次の会話の要点を200文字以内で要約してください。\n\n"
    for q, a in recent:
        prompt += f"ユーザー: {q}\nAI: {a}\n"
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"要約生成に失敗しました: {e}")
        return ""

# 回答生成
def generate_response_with_history(user_q, chat_log, ref_q, ref_a):
    # systemプロンプト
    system_prompt = (
        "あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。"
        "以下のFAQを参考に200文字以内で回答してください。\n"
        f"FAQ質問: {ref_q}\nFAQ回答: {ref_a}"
    )
    # 過去の会話履歴をmessagesに追加
    messages = [{"role": "system", "content": system_prompt}]
    # 古い順に並べる
    for q, a in reversed(chat_log[-5:]):  # 直近5ターン分
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    # 今回の質問
    messages.append({"role": "user", "content": user_q})

    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"AI回答生成に失敗しました: {e}")
        return "申し訳ありません、AIによる回答生成に失敗しました。"
# セッションステート初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# 入力フォーム
with st.form(key="chat_form", clear_on_submit=True):
    user_q = st.text_input("質問をどうぞ：")
    send = st.form_submit_button("送信")

# チャット送信処理
if send and user_q:
    if not is_valid_input(user_q):
        st.warning("入力が不正です。3〜300文字、記号率30%未満にしてください。")
    else:
        ref_q, ref_a = find_top_similar(user_q, faq_df)
        if ref_q is None:
            answer = "申し訳ありません、関連FAQが見つかりませんでした。"
        else:
            with st.spinner("回答生成中…"):
                answer = generate_response_with_history(
                    user_q,
                    st.session_state.chat_log,
                    ref_q,
                    ref_a
                )
                
        # 履歴に追加 & 保存処理
        st.session_state.chat_log.insert(0, (user_q, answer))
        append_to_csv(user_q, answer)
        append_to_gsheet(user_q, answer)

        # 履歴の最大件数制御
        if len(st.session_state.chat_log) > max_log:
            st.session_state.chat_log = st.session_state.chat_log[:max_log]
        st.experimental_rerun()

# チャット履歴表示
if st.session_state.chat_log:
    st.subheader(" チャット履歴")
    logs = st.session_state.chat_log if log_order == "新しい順" else list(reversed(st.session_state.chat_log))
    for q, a in logs:
        st.markdown(
            f'<div class="chat-text"><b>🧑‍💻 質問:</b> {q}<br><b>🤖 回答:</b> {a}</div><hr>',
            unsafe_allow_html=True
        )
