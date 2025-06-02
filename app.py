import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, random, re, unicodedata
import base64

# ──────────────────────────────
# ページ設定
# ──────────────────────────────
st.set_page_config(page_title="LRADサポートチャット", layout="centered")

# ──────────────────────────────
# OpenAIキー
# ──────────────────────────────
openai.api_key = st.secrets.OpenAIAPI.openai_api_key

def inject_custom_css(body_font_size: str = "16px"):
    title_font_size = f"calc({body_font_size} * 1.6)"
    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            font-size: {body_font_size} !important;
        }}
        div[data-testid="stMarkdownContainer"] h1 {{
            font-size: {title_font_size} !important;
            line-height: 1.4;
        }}
        div[data-testid="stMarkdownContainer"] h2 {{
            font-size: calc({title_font_size} * 0.8) !important;
        }}
        div[data-testid="stMarkdownContainer"] h3 {{
            font-size: calc({title_font_size} * 0.7) !important;
        }}
        p > small {{
            font-size: calc({body_font_size} * 0.9) !important;
        }}
        div[data-testid="text-input-label"] > div {{
            font-size: {body_font_size} !important;
        }}
        input[type="text"], input[type="text"]::placeholder {{
            font-size: {body_font_size} !important;
        }}
        button[kind], span, label {{
            font-size: {body_font_size} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ──────────────────────────────
# ユーティリティ
# ──────────────────────────────
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    res = openai.embeddings.create(input=[text], model=model)
    return np.array(res.data[0].embedding)

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

# ──────────────────────────────
# CSV読込み
# ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_faq_all(path="faq_all.csv", cached="faq_all_with_embed.csv"):
    if os.path.exists(cached):
        df = pd.read_csv(cached)
        df["embedding"] = df["embedding"].apply(eval).apply(np.array)
    else:
        df = pd.read_csv(path)
        with st.spinner("全FAQへ埋め込み計算中…（初回のみ）"):
            df["embedding"] = df["質問"].apply(get_embedding)
        df.to_csv(cached, index=False)
    return df

@st.cache_data(show_spinner=False)
def load_faq_common(path="faq_common.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")  # ←ここ重要！
    df.columns = df.columns.str.strip()  # ← 列名の空白除去
    return df

faq_df = load_faq_all()
common_faq_df = load_faq_common()

# ──────────────────────────────
# ランダムFAQ表示
# ──────────────────────────────
def display_random_common_faqs(common_faq_df, n=3):
    sampled = common_faq_df.sample(n)
    for i, row in enumerate(sampled.itertuples(), 1):
        question = getattr(row, "質問", "（質問が不明です）")
        answer = getattr(row, "回答", "（回答が不明です）")
        st.markdown(f"**❓ {row[1]}**")  # row[0] = 質問
        st.markdown(f"🅰️ {row[2]}")
        st.markdown("---")


# ──────────────────────────────
# 類似質問検索
# ──────────────────────────────
def find_top_similar(q, df, k=1):
    if len(q.strip()) < 2:
        return None, None
    q_vec = get_embedding(q)
    faq_vecs = np.stack(df["embedding"].to_numpy())
    sims = cosine_similarity([q_vec], faq_vecs)[0]
    idx = sims.argsort()[::-1][:k][0]
    return df.iloc[idx]["質問"], df.iloc[idx]["回答"]

# ──────────────────────────────
# 回答生成
# ──────────────────────────────
def generate_response(user_q, ref_q, ref_a):
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
# セッションステート
# ──────────────────────────────
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# ──────────────────────────────
# UI描画　LRAD装置画像あり
# ──────────────────────────────
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

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

# サイドバーで文字サイズを選択
st.sidebar.title("⚙️ 表示設定")
font_size = st.sidebar.selectbox("文字サイズを選んでください", ["小", "中", "大"])

# ❶ サイドバー選択肢 & マップ
font_size_map = {"小": "14px", "中": "18px", "大": "24px"}
img_width_map = {"小": 60, "中": 80, "大": 110}   # ← 好みで調整

selected_font = font_size_map[font_size]
selected_img  = img_width_map[font_size]

inject_custom_css(selected_font)

# よくある質問（CSV② からランダム）
st.markdown("### 💡 よくある質問（ランダム表示）")
display_random_common_faqs(common_faq_df, n=3) 

st.divider()

# 入力フォーム
with st.form(key="chat_form", clear_on_submit=True):
    user_q = st.text_input("質問をどうぞ：")
    send = st.form_submit_button("送信")

if send and user_q:
    if not is_valid_input(user_q):
        st.warning("入力が不正です。3〜300文字、記号率30%未満にしてください。")
    else:
        ref_q, ref_a = find_top_similar(user_q, faq_df)
        if ref_q is None:
            answer = "申し訳ありません、関連FAQが見つかりませんでした。"
        else:
            with st.spinner("回答生成中…"):
                answer = generate_response(user_q, ref_q, ref_a)
        st.session_state.chat_log.insert(0, (user_q, answer))
        st.experimental_rerun()

# チャット履歴
if st.session_state.chat_log:
    st.subheader("📜 チャット履歴")
    for q, a in st.session_state.chat_log:
        st.markdown(f"**🧑‍💻 質問:** {q}")
        st.markdown(f"**🤖 回答:** {a}")
        st.markdown("---")
