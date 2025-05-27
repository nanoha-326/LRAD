import streamlit as st
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import re
import unicodedata

# --- Streamlitの設定 ---
st.set_page_config(page_title="LRADサポートチャット", page_icon="\U0001F4D8", layout="centered")

# --- OpenAI APIキー設定 ---
client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)

# --- 埋め込みモデル読み込み ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- システムプロンプト ---
system_prompt = """
あなたはLRAD専用のチャットボットです。
「LRAD（エルラド）」とは熱分解装置（遠赤外線電子熱分解装置）のことで、これは有機廃棄物の処理装置です。
あなたの役割は、この装置の検証をサポートすることです。

以下の点を守ってください：
・あなたはLRADの専門家として利用者の質問にわかりやすく回答し、処理検証をサポートできます。
・装置に関連することのみを答えてください。それ以外の質問（例：天気、有名人、趣味、思想、料理、政治、ゲーム、スポーツ、健康など）には絶対に答えないでください。
・世間話をされてもLRADに関係のない場合は答えないでください。
・質問には親切に、できるだけ分かりやすく答えてください。
・FAQのファイル内に類似する情報がない場合は、回答が不明であることを丁寧に伝え、適切に対応してください。
"""

# --- 入力バリデーション ---
def is_valid_input(text: str) -> bool:
    text = text.strip()
    if len(text) < 3 or len(text) > 300:
        return False
    non_alpha_ratio = len(re.findall(r'[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]', text)) / len(text)
    if non_alpha_ratio > 0.3:
        return False
    try:
        normalized = unicodedata.normalize('NFKC', text)
        normalized.encode('utf-8')
    except UnicodeError:
        return False
    return True

# --- FAQデータ読み込み ---
@st.cache_data
def load_faq(csv_file):
    df = pd.read_csv(csv_file)
    df['embedding'] = df['質問'].apply(lambda x: model.encode(x))
    return df

faq_df = load_faq("faq.csv")

# --- 類似質問検索 ---
def find_similar_question(user_input, faq_df):
    user_vec = model.encode([user_input])
    faq_vecs = list(faq_df['embedding'])
    scores = cosine_similarity(user_vec, faq_vecs)[0]
    top_idx = scores.argmax()
    return faq_df.iloc[top_idx]['質問'], faq_df.iloc[top_idx]['回答']

# --- GPTによる補完応答 ---
def generate_response(context_question, context_answer, user_input):
    prompt = (
        f"以下はFAQに基づいたチャットボットの会話です。\n\n"
        f"質問: {context_question}\n回答: {context_answer}\n\n"
        f"ユーザーの質問: {user_input}\n\n"
        "これを参考に、丁寧でわかりやすく自然な回答をしてください。"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=1.5
    )
    return response.choices[0].message['content']

# --- チャットログ保存 ---
def save_log(log_data):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chatlog_{now}.csv"
    log_df = pd.DataFrame(log_data, columns=["ユーザーの質問", "チャットボットの回答"])
    log_df.to_csv(filename, index=False)
    return filename

# --- UIタイトル ---
st.title("LRADサポートチャット")
st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

# --- セッションステート初期化 ---
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

# --- サイドバー設定 ---
with st.sidebar:
    st.markdown("### ⚙️ 表示設定")
    font_size = st.radio("文字サイズ", ["小", "標準", "大"], index=1)
    st.divider()
    st.markdown("背景色などの切り替え機能も追加できます")

font_size_map = {"小": "14px", "標準": "16px", "大": "20px"}

# --- CSSでLINE風吹き出しデザイン ---
st.markdown(f"""
<style>
body {{
    background-color: #f6f6f6;
}}
.chat-container {{
    max-width: 700px;
    margin: 0 auto 100px auto;
    padding: 10px;
}}
.user-message {{
    background-color: #dcf8c6;
    padding: 12px 15px;
    border-radius: 20px 20px 0 20px;
    margin: 10px 0 10px 40px;
    max-width: 75%;
    font-size: {font_size_map[font_size]};
    word-break: break-word;
    box-shadow: 0 1px 1px rgb(0 0 0 / 0.1);
}}
.bot-message {{
    background-color: #ffffff;
    padding: 12px 15px;
    border-radius: 20px 20px 20px 0;
    margin: 10px 40px 10px 0;
    max-width: 75%;
    font-size: {font_size_map[font_size]};
    word-break: break-word;
    box-shadow: 0 1px 1px rgb(0 0 0 / 0.1);
}}
.input-area {{
    position: fixed;
    bottom: 0;
    width: 100%;
    max-width: 720px;
    background-color: #ffffff;
    padding: 10px;
    box-shadow: 0 -1px 5px rgb(0 0 0 / 0.1);
    display: flex;
    gap: 8px;
    box-sizing: border-box;
    border-top: 1px solid #ddd;
    margin: 0 auto;
}}
.input-area input[type="text"] {{
    flex-grow: 1;
    font-size: 16px;
    padding: 10px;
    border-radius: 20px;
    border: 1px solid #ccc;
}}
.input-area button {{
    background-color: #0078FF;
    border: none;
    color: white;
    padding: 10px 18px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: bold;
}}
.input-area button:hover {{
    background-color: #005bb5;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# --- チャットログ表示（最新が下）---
for user_msg, bot_msg in reversed(st.session_state.chat_log):
    st.markdown(f'<div class="user-message">{user_msg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-message">{bot_msg}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- チャット入力フォーム ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="質問をどうぞ", key="user_input")
    submitted = st.form_submit_button("送信")

    if submitted and user_input:
        if not is_valid_input(user_input):
            error_message = "エラーが発生しています。時間を空けてから再度お試しください。"
            st.session_state.chat_log.append((user_input, error_message))
            st.experimental_rerun()

        similar_q, similar_a = find_similar_question(user_input, faq_df)
        final_response = generate_response(similar_q, similar_a, user_input)
        st.session_state.chat_log.append((user_input, final_response))
        st.experimental_rerun()

# --- チャットログ保存ボタン ---
if st.button("チャットログを保存"):
    filename = save_log(st.session_state.chat_log)
    st.success(f"チャットログを保存しました: {filename}")
    with open(filename, "rb") as f:
        st.download_button(
            label="このチャットログをダウンロード",
            data=f,
            file_name=filename,
            mime="text/csv"
        )
