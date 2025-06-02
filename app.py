import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- API KEY ---
openai.api_key = st.secrets.OpenAIAPI.openai_api_key 

# --- OpenAI 埋め込み取得 ---
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

# --- FAQ 読み込み（埋め込み自動付与） ---
@st.cache_data
def load_faq(path="faq.csv", embed_path="faq_with_embeddings.csv"):
    if os.path.exists(embed_path):
        df = pd.read_csv(embed_path)
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    else:
        df = pd.read_csv(path)
        with st.spinner("FAQにembeddingを付与中...（初回のみ）"):
            df["embedding"] = df["質問"].apply(get_embedding)
        df.to_csv(embed_path, index=False)
    return df

faq_df = load_faq()

# --- 類似質問を検索 ---
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
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- UI ---
st.set_page_config(page_title="LRADサポートチャット", layout="centered")
st.title("🤖 LRADサポートチャット")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# --- 入力フォーム ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("質問をどうぞ：", key="user_input")
    submitted = st.form_submit_button("送信")

# --- 検索候補表示 ---
if user_input:
    st.subheader("🔍 入力に基づくおすすめの質問")

    suggested_qas = find_top_similar_questions(user_input, faq_df)
    for i, (q, a) in enumerate(suggested_qas):
        if st.button(f"{i+1}. {q}"):
            with st.spinner("回答生成中…お待ちください。"):
                answer = generate_response(q, a, q)
            st.session_state.chat_log.insert(0, (q, answer))
            st.rerun()

# --- ユーザーから送信された場合の処理 ---
if submitted and user_input:
    with st.spinner("回答生成中…お待ちください。"):
        suggested_qas = find_top_similar_questions(user_input, faq_df, top_n=1)
        if suggested_qas:
            matched_q, matched_a = suggested_qas[0]
        else:
            matched_q, matched_a = "該当なし", "申し訳ありませんが、該当するFAQが見つかりませんでした。"
        answer = generate_response(user_input, matched_a, matched_q)
    st.session_state.chat_log.insert(0, (user_input, answer))
    st.session_state.user_input = ""

# --- チャットログ表示 ---
st.subheader("📜 チャット履歴")
for q, a in st.session_state.chat_log:
    st.markdown(f"**🧑‍💻 質問:** {q}")
    st.markdown(f"**🤖 回答:** {a}")
    st.markdown("---")
