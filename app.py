import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, random, re, unicodedata, json
import base64

# ページ設定
st.set_page_config(page_title="LRADサポートチャット", layout="centered")

# OpenAIキー
client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)

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

# Embedding取得
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    res = client.embeddings.create(input=[text], model=model)
    return res.data[0].embedding

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

# CSV読み込み
@st.cache_data(show_spinner=False)
def load_faq_all(path="faq_all.csv", cached="faq_all_with_embed.csv"):
    if os.path.exists(cached):
        df = pd.read_csv(cached)
        try:
            df["embedding"] = df["embedding"].apply(json.loads).apply(np.array)
        except Exception as e:
            st.error(f"埋め込みの読み込みに失敗しました: {e}")
            st.stop()
    else:
        df = pd.read_csv(path)
        with st.spinner("全FAQへ埋め込み計算中…（初回のみ）"):
            df["embedding"] = df["質問"].apply(get_embedding)
        # 文字列化して保存
        df["embedding"] = df["embedding"].apply(lambda x: json.dumps(x.tolist()))
        df.to_csv(cached, index=False)
        # 読み込み直し
        df["embedding"] = df["embedding"].apply(json.loads).apply(np.array)
    return df

@st.cache_data(show_spinner=False)
def load_faq_common(path="faq_common.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

faq_df = load_faq_all()
common_faq_df = load_faq_common()

# FAQ表示
def display_random_common_faqs(common_faq_df, n=3):
    sampled = common_faq_df.sample(n)
    for i, row in enumerate(sampled.itertuples(), 1):
        question = getattr(row, "質問", "（質問が不明です）")
        answer = getattr(row, "回答", "（回答が不明です）")
        st.markdown(
            f'<div class="chat-text"><b>❓ {question}</b><br>🅰️ {answer}</div><hr>',
            unsafe_allow_html=True
        )

# 類似質問検索
def find_top_similar(q, df, k=1):
    if len(q.strip()) < 2:
        return None, None
    q_vec = get_embedding(q)
    faq_vecs = np.stack(df["embedding"].to_numpy())
    sims = cosine_similarity([q_vec], faq_vecs)[0]
    idx = sims.argsort()[::-1][:k][0]
    return df.iloc[idx]["質問"], df.iloc[idx]["回答"]

# 回答生成
def generate_response(user_q, ref_q, ref_a):
    prompt = (
        "あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。\n"
        "以下のFAQを参考に200文字以内で回答してください。\n\n"
        f"FAQ質問: {ref_q}\nFAQ回答: {ref_a}\n\nユーザー質問: {user_q}"
    )
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return res.choices[0].message.content.strip()

# セッションステート初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# サイドバー設定
st.sidebar.title("⚙️ 表示設定")
font_size = st.sidebar.selectbox("文字サイズを選んでください", ["小", "中", "大"])
font_size_map = {"小": "14px", "中": "18px", "大": "24px"}
img_width_map = {"小": 60, "中": 80, "大": 110}

selected_font = font_size_map[font_size]
selected_img = img_width_map[font_size]

inject_custom_css(selected_font)

# ヘッダー画像
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_base64_image("LRADimg.png")

st.markdown(
    f"""
    <div style="display:flex; align-items:center;" class="chat-header">
        <img src="data:image/png;base64,{image_base64}"
             width="80px" style="margin-right:10px;">
        <h1 style="margin:0; font-size:40px; font-weight:bold;">LRADサポートチャット</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

# よくある質問表示
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

# チャット履歴表示
if st.session_state.chat_log:
    st.subheader("📜 チャット履歴")
    for q, a in st.session_state.chat_log:
        st.markdown(
            f'<div class="chat-text"><b>🧑‍💻 質問:</b> {q}<br><b>🤖 回答:</b> {a}</div><hr>',
            unsafe_allow_html=True
        )
