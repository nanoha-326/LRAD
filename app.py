import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

# --- ページ設定 ---
st.set_page_config(page_title="LRADサポートチャット", layout="centered")

# --- APIキー ---
openai.api_key = st.secrets.OpenAIAPI.openai_api_key 

# --- 埋め込み取得関数 ---
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

# --- 全質問用FAQ読み込み（回答検索用） ---
@st.cache_data
def load_faq_all(csv_file):
    df = pd.read_csv(csv_file)
    df['embedding'] = df['質問'].apply(lambda x: get_embedding(x))
    return df

# --- よくある質問用FAQ読み込み（ランダム表示用） ---
@st.cache_data
def load_faq_common(csv_file):
    df = pd.read_csv(csv_file)
    return df

faq_df = load_faq_all("faq_all.csv")         # 全FAQ（embedding付き）
common_faq_df = load_faq_common("faq_common.csv")  # よくある質問（embeddingなし）

# --- ランダムによくある質問3件を表示 ---
def display_random_common_faqs(common_faq_df, n=3):
    sampled = common_faq_df.sample(n)
    st.markdown("### よくある質問の例")
    for i, row in enumerate(sampled.itertuples(), 1):
        st.markdown(f"**{i}. {row.質問}**")
        st.markdown(f"回答: {row.回答}")
        st.markdown("---")

# --- 類似質問検索 ---
def find_top_similar_questions(user_input, faq_df, top_n=5):
    if len(user_input.strip()) < 2:
        return []
    user_vec = get_embedding(user_input)
    faq_vecs = np.stack(faq_df['embedding'].to_numpy())
    scores = cosine_similarity([user_vec], faq_vecs)[0]
    top_indices = scores.argsort()[::-1][:top_n]
    return faq_df.iloc[top_indices][['質問', '回答']].values.tolist()

# --- 回答生成 ---
def generate_response(user_input, matched_answer, matched_question):
    prompt = f"""あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。
次のFAQと照らし合わせて、200文字以内で質問に回答してください。

質問: {user_input}
最も近いFAQ: {matched_question}
回答: {matched_answer}
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- セッションステート初期化 ---
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- UI ---
st.title("🤖 LRADサポートチャット")

# 入力欄（即時反応）
user_input = st.text_input("質問をどうぞ：", value=st.session_state.user_input, key="user_input")

# --- 既存のフォームの直前にランダムFAQ表示を入れる例 ---
display_random_common_faqs(common_faq_df, n=3)

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("質問をどうぞ：", key="user_input")
    submitted = st.form_submit_button("送信")

    if submitted and user_input:
        if not is_valid_input(user_input):
            st.session_state.chat_log.insert(0, (user_input, "エラー：入力が不正です。"))
            st.experimental_rerun()
        similar_q, similar_a = find_similar_question(user_input, faq_df)
        with st.spinner("回答生成中…お待ちください。"):
            answer = generate_response(similar_q, similar_a, user_input)
        st.session_state.chat_log.insert(0, (user_input, answer))
        st.experimental_rerun()

# チャット履歴表示
st.subheader("📜 チャット履歴")
for q, a in st.session_state.chat_log:
    st.markdown(f"**🧑‍💻 質問:** {q}")
    st.markdown(f"**🤖 回答:** {a}")
    st.markdown("---")
