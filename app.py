import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, re, unicodedata, base64

# ──────────────────────────────
# ページ設定 & OpenAI キー
# ──────────────────────────────
st.set_page_config(page_title="LRADサポートチャット", layout="centered")
openai.api_key = st.secrets.OpenAIAPI.openai_api_key

# ──────────────────────────────
# 1️⃣ カスタム CSS（本文とタイトルを別サイズで）
# ──────────────────────────────

def inject_custom_css(body_px: str, title_px: str):
    """選択された本文とタイトルのフォントサイズを全体へ注入"""
    st.markdown(f"""
    <style>
    /* 本文フォント */
    html, body, .stApp {{font-size:{body_px}!important;}}

    /* タイトル (st.title → h1)*/
    div[data-testid="stMarkdownContainer"] h1 {{
        font-size:{title_px}!important; line-height:1.4;}}

    /* 見出し h2/h3 はタイトル比 */
    div[data-testid="stMarkdownContainer"] h2 {{
        font-size:calc({title_px}*0.8)!important;}}
    div[data-testid="stMarkdownContainer"] h3 {{
        font-size:calc({title_px}*0.7)!important;}}

    /* caption (<p><small>) */
    p>small {{font-size:calc({body_px}*0.9)!important;}}

    /* 入力ラベル・入力文字・プレースホルダー */
    div[data-testid="text-input-label"]>div, input[type="text"],
    input[type="text"]::placeholder {{font-size:{body_px}!important;}}

    /* ボタン文字など */
    button[kind], span, label {{font-size:{body_px}!important;}}
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────
# 2️⃣ ユーティリティ
# ──────────────────────────────

def get_embedding(text: str, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    res = openai.embeddings.create(input=[text], model=model)
    return np.array(res.data[0].embedding)


def is_valid_input(text: str) -> bool:
    text = text.strip()
    if not (3 <= len(text) <= 300):
        return False
    if len(re.findall(r"[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]", text)) / len(text) > 0.3:
        return False
    try:
        unicodedata.normalize("NFKC", text).encode("utf-8")
    except UnicodeError:
        return False
    return True

# ──────────────────────────────
# 3️⃣ CSV 読み込み
# ──────────────────────────────

@st.cache_data(show_spinner=False)
def load_faq_all(path="faq_all.csv", cached="faq_all_with_embed.csv"):
    if os.path.exists(cached):
        df = pd.read_csv(cached)
        df["embedding"] = df["embedding"].apply(eval).apply(np.array)
    else:
        df = pd.read_csv(path)
        with st.spinner("FAQ の埋め込みベクトル計算中…（初回のみ）"):
            df["embedding"] = df["質問"].apply(get_embedding)
        df.to_csv(cached, index=False)
    return df

@st.cache_data(show_spinner=False)
def load_faq_common(path="faq_common.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

faq_df = load_faq_all()
common_faq_df = load_faq_common()

# ──────────────────────────────
# 4️⃣ サイドバー：文字サイズ設定
# ──────────────────────────────

st.sidebar.title("⚙️ 表示設定")
size_choice = st.sidebar.selectbox("文字サイズを選択", ["小", "中", "大"], index=1)
body_map = {"小": "14px", "中": "18px", "大": "22px"}
img_map  = {"小": 60, "中": 80, "大": 110}

body_px = body_map[size_choice]
# タイトルは本文の1.6倍
title_px = str(int(body_px.replace("px", ""))*16//10)+"px"
logo_w   = img_map[size_choice]

inject_custom_css(body_px, title_px)

# ──────────────────────────────
# 5️⃣ ヘッダー（ロゴ+タイトル）
# ──────────────────────────────

def get_base64_image(path: str):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = get_base64_image("LRADimg.png")

st.markdown(f"""
<div style="display:flex;align-items:center;" class="chat-text">
  <img src="data:image/png;base64,{logo_b64}" width="{logo_w}" style="margin-right:10px;"/>
  <h1 style="margin:0;">LRADサポートチャット</h1>
</div>
""", unsafe_allow_html=True)

st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

# ──────────────────────────────
# 6️⃣ よくある質問をランダム表示
# ──────────────────────────────

def show_random_faq(df, n=3):
    for _, row in df.sample(n=min(n, len(df))).iterrows():
        st.markdown(f"<div class='chat-text'><b>❓ {row['質問']}</b><br>🅰️ {row['回答']}</div><hr>", unsafe_allow_html=True)

st.markdown("### 💡 よくある質問（ランダム表示）")
show_random_faq(common_faq_df, 3)

st.divider()

# ──────────────────────────────
# 7️⃣ 類似質問検索 & 回答生成
# ──────────────────────────────

def search_similar(user_q: str):
    if len(user_q.strip()) < 2:
        return None, None
    u_vec = get_embedding(user_q)
    mat   = np.stack(faq_df["embedding"].to_numpy())
    sims  = cosine_similarity([u_vec], mat)[0]
    idx   = sims.argmax()
    return faq_df.iloc[idx]["質問"], faq_df.iloc[idx]["回答"]


def answer_by_gpt(user_q: str, ref_q: str, ref_a: str):
    prompt = (
        "あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。\n"
        "以下のFAQを参考に200文字以内で回答してください。\n\n"
        f"FAQ質問: {ref_q}\nFAQ回答: {ref_a}\n\nユーザー質問: {user_q}"
    )
    res = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return res.choices[0].message.content.strip()

# ──────────────────────────────
# 8️⃣ 入力フォーム & 応答
# ──────────────────────────────
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

with st.form(key="chat_form", clear_on_submit=True):
    user_q = st.text_input("質問をどうぞ：")
    send   = st.form_submit_button("送信")

if send and user_q:
    if not is_valid_input(user_q):
        st.warning("入力が不正です。3〜300文字、記号率30%未満にしてください。")
    else:
        ref_q, ref_a = search_similar(user_q)
        if ref_q is None:
            answer = "申し訳ありません、関連FAQが見つかりませんでした。"
        else:
            with st.spinner("回答生成中…"):
                answer = answer_by_gpt(user_q, ref_q, ref_a)
        st.session_state.chat_log.insert(0, (user_q, answer))
        st.experimental_rerun()

# ──────────────────────────────
# 9️⃣ チャット履歴
# ──────────────────────────────
if st.session_state.chat_log:
    st.subheader("📜 チャット履歴")
    for q, a in st.session_state.chat_log:
        st.markdown(f"<div class='chat-text'><b>🧑‍💻 質問:</b> {q}<br><b>🤖 回答:</b> {a}</div><hr>", unsafe_allow_html=True)
