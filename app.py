import streamlit as st
import pandas as pd
import numpy as np
import datetime
import re
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

st.set_page_config(page_title="LRADサポートチャット", page_icon="📘", layout="centered")

client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)

system_prompt = """
あなたはLRAD専用のチャットボットです。
「LRAD（エルラド）」とは熱分解装置（遠赤外線電子熱分解装置）のことで、これは有機廃棄物の処理装置です。
あなたの役割は、この装置の検証をサポートすることです。

以下の点を守ってください：
・装置に関連することのみを答えてください。
・関係ない話題（天気、芸能、スポーツなど）には答えないでください。
・FAQにない場合は「わかりません」と丁寧に答えてください。
"""

def is_valid_input(text: str) -> bool:
    text = text.strip()
    if len(text) < 3 or len(text) > 300:
        return False
    non_alpha_ratio = len(re.findall(r'[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]', text)) / len(text)
    if non_alpha_ratio > 0.3:
        return False
    try:
        unicodedata.normalize('NFKC', text).encode('utf-8')
    except UnicodeError:
        return False
    return True

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding  # numpyにしないでlistのまま返す

@st.cache_data
def load_faq(csv_file):
    df = pd.read_csv(csv_file)
    df['embedding'] = df['質問'].apply(lambda x: get_embedding(x))
    return df

faq_df = load_faq("faq.csv")

def find_similar_question(user_input, faq_df):
    user_vec = get_embedding(user_input)
    faq_vecs = np.array(faq_df['embedding'].tolist())  # listのリスト→numpy配列
    scores = cosine_similarity([user_vec], faq_vecs)[0]
    top_idx = scores.argmax()
    return faq_df.iloc[top_idx]['質問'], faq_df.iloc[top_idx]['回答']

def generate_response(context_q, context_a, user_input):
    prompt = f"以下はFAQに基づいたチャットボットの会話です。\n\n質問: {context_q}\n回答: {context_a}\n\nユーザーの質問: {user_input}\n\nこれを参考に、丁寧でわかりやすく自然な回答をしてください。"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=1.2
    )
    return response.choices[0].message.content

def save_log(log_data):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chatlog_{now}.csv"
    pd.DataFrame(log_data, columns=["ユーザーの質問", "チャットボットの回答"]).to_csv(filename, index=False)
    return filename

st.title("LRADサポートチャット")
st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

st.title("LRADサポートチャット")

# ページ上部にCSS埋め込み
st.markdown(f"""
    <style>
    /* ここにCSS */
    input[type="text"], textarea {{
        font-size: {font_px}px !important;
    }}
    button {{
        font-size: {font_px}px !important;
    }}
    div.stChatMessage p {{
        font-size: {font_px}px !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")


if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

if st.button("チャットログを保存"):
    filename = save_log(st.session_state.chat_log)
    st.success(f"チャットログを保存しました: {filename}")
    with open(filename, "rb") as f:
        st.download_button("このチャットログをダウンロード", data=f, file_name=filename, mime="text/csv")

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

for user_msg, bot_msg in st.session_state.chat_log:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
