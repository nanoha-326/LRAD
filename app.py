import streamlit as st
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import re
import unicodedata

# ---------- Streamlit の設定 ----------
st.set_page_config(
    page_title="LRADサポートチャット",
    page_icon="📘",
    layout="centered"
)

# ---------- OpenAI API キー ----------
openai.api_key = st.secrets.OpenAIAPI.openai_api_key   # ← V1 でもこの書き方で OK

# ---------- 埋め込みモデル ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- システムプロンプト ----------
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

# ---------- 入力バリデーション ----------
def is_valid_input(text: str) -> bool:
    text = text.strip()
    if len(text) < 3 or len(text) > 300:
        return False
    non_alpha_ratio = len(re.findall(r"[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]", text)) / len(text)
    if non_alpha_ratio > 0.3:
        return False
    try:
        unicodedata.normalize("NFKC", text).encode("utf-8")
    except UnicodeError:
        return False
    return True


# ---------- FAQ 読み込み ----------
@st.cache_data(show_spinner="FAQ 読み込み中…")
def load_faq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["embedding"] = df["質問"].apply(lambda x: model.encode(x))
    return df

faq_df = load_faq("faq.csv")


# ---------- 類似質問検索 ----------
def find_similar_question(query: str):
    user_vec = model.encode([query])
    faq_vecs = list(faq_df["embedding"])
    scores = cosine_similarity(user_vec, faq_vecs)[0]
    top_idx = scores.argmax()
    return faq_df.iloc[top_idx]["質問"], faq_df.iloc[top_idx]["回答"]


# ---------- GPT で回答生成 ----------
def generate_response(context_q: str, context_a: str, user_input: str) -> str:
    prompt = (
        "以下はFAQに基づいたチャットボットの会話です。\n\n"
        f"質問: {context_q}\n回答: {context_a}\n\n"
        f"ユーザーの質問: {user_input}\n\n"
        "これを参考に、丁寧でわかりやすく自然な回答をしてください。"
    )
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=1.5,
    )
    return res.choices[0].message.content


# ---------- チャットログ保存 ----------
def save_log(log):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"chatlog_{ts}.csv"
    pd.DataFrame(log, columns=["ユーザー", "チャットボット"]).to_csv(fname, index=False)
    return fname


# ---------- UI ----------
st.title("LRADサポートチャット")
st.caption("※FAQとGPTを用いて回答します。内容の正確性は保証されません。")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

with st.sidebar:
    st.markdown("### ⚙️ 表示設定")
    font_size = st.radio("文字サイズ", ["小", "標準", "大"], index=1)
    st.divider()
    st.markdown("背景色テーマ等を追加予定")

font_size_map = {"小": "14px", "標準": "16px", "大": "20px"}

# ---------- CSS（LINE風） ----------
st.markdown(
    f"""
<style>
body {{ background-color:#f6f6f6; }}

.chat-container {{ max-width:700px;margin:0 auto 100px;padding:10px; }}

.user-message {{
    background:#dcf8c6;
    border-radius:20px 20px 0 20px;
    margin:10px 0 10px 40px;
    padding:12px 15px;
    max-width:75%;
    font-size:{font_size_map[font_size]};
    word-break:break-word;
    box-shadow:0 1px 1px rgba(0,0,0,.1);
}}
.bot-message {{
    background:#fff;
    border-radius:20px 20px 20px 0;
    margin:10px 40px 10px 0;
    padding:12px 15px;
    max-width:75%;
    font-size:{font_size_map[font_size]};
    word-break:break-word;
    box-shadow:0 1px 1px rgba(0,0,0,.1);
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# ログ表示（新しい順を下に）
for u, a in st.session_state.chat_log:
    st.markdown(f'<div class="user-message">{u}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-message">{a}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- 入力欄（画面下固定ではなく通常フォーム） ----------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="質問をどうぞ")
    submitted = st.form_submit_button("送信")

    if submitted and user_input:
        if not is_valid_input(user_input):
            st.session_state.chat_log.append(
                (user_input, "入力エラー：3〜300文字で、記号を多用しないでください。")
            )
            st.experimental_rerun()

        similar_q, similar_a = find_similar_question(user_input)
        answer = generate_response(similar_q, similar_a, user_input)
        st.session_state.chat_log.append((user_input, answer))
        st.experimental_rerun()

# ---------- チャットログ保存 ----------
if st.button("チャットログを保存"):
    fname = save_log(st.session_state.chat_log)
    st.success(f"チャットログを保存しました：{fname}")
    with open(fname, "rb") as f:
        st.download_button("ダウンロード", data=f, file_name=fname, mime="text/csv")
