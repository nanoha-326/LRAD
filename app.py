import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, random, re, unicodedata

# ──────────────────────────────
# ページ設定
# ──────────────────────────────
st.set_page_config(page_title="LRADサポートチャット", layout="centered")

# ──────────────────────────────
# OpenAIキー
# ──────────────────────────────
openai.api_key = st.secrets.OpenAIAPI.openai_api_key

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
    return pd.read_csv(path)

faq_df = load_faq_all()
common_faq_df = load_faq_common()

# ──────────────────────────────
# ランダムFAQ表示
# ──────────────────────────────
def show_random_faq(df, n=3):
    n = min(n, len(df))
    for i, row in df.sample(n).itertuples(index=False, name=None):
        st.markdown(f"**❓ {row[0]}**")  # row[0] = 質問
        st.markdown(f"🅰️ {row[1]}")
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
# UI描画
# ──────────────────────────────
st.title("🤖 LRADサポートチャット")

# よくある質問（CSV② からランダム）
st.markdown("### 💡 よくある質問（ランダム表示）")
show_random_faq(common_faq_df, n=3)

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
