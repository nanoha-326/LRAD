import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, re, unicodedata, base64

# ── ページ設定 ────────────────────────────────────
st.set_page_config(page_title="LRADサポートチャット", layout="centered")
openai.api_key = st.secrets.OpenAIAPI.openai_api_key

# ── CSS（本文フォントサイズだけ可変） ─────────────
def inject_custom_css(px: str):
    st.markdown(f"""
    <style>
    .chat-text, .stCaption, .stTextInput > label {{font-size:{px}!important}}
    .stTextInput input {{font-size:{px}!important}}
    ::placeholder {{font-size:{px}!important}}
    </style>""", unsafe_allow_html=True)

# ── ユーティリティ ───────────────────────────────
def is_valid_input(text:str)->bool:
    text=text.strip()
    if not (3<=len(text)<=300): return False
    if len(re.findall(r'[^A-Za-z0-9ぁ-んァ-ヶ一-龠\\s]',text))/len(text)>0.3: return False
    try: unicodedata.normalize("NFKC",text).encode("utf-8")
    except UnicodeError: return False
    return True

def get_embedding(t, model="text-embedding-3-small"):
    t=t.replace("\n"," ")
    return np.array(openai.embeddings.create(input=[t], model=model).data[0].embedding)

@st.cache_data(show_spinner=False)
def load_faq_all(path="faq_all.csv", cached="faq_all_with_embed.csv"):
    if os.path.exists(cached):
        df=pd.read_csv(cached)
        df["embedding"]=df["embedding"].apply(eval).apply(np.array)
    else:
        df=pd.read_csv(path)
        with st.spinner("FAQ埋め込み計算中…"):
            df["embedding"]=df["質問"].apply(get_embedding)
        df.to_csv(cached,index=False)
    return df

@st.cache_data(show_spinner=False)
def load_faq_common(path="faq_common.csv"):
    df=pd.read_csv(path,encoding="utf-8-sig"); df.columns=df.columns.str.strip(); return df

faq_df=load_faq_all()
common_faq_df=load_faq_common()

def find_top_similar(q):
    q_vec=get_embedding(q)
    mat=np.stack(faq_df["embedding"].to_numpy())
    sims=np.dot(mat, q_vec)/(np.linalg.norm(mat,1)+1e-9)
    idx=sims.argmax()
    return faq_df.iloc[idx]["質問"], faq_df.iloc[idx]["回答"]

def gpt_chat(messages):
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    ).choices[0].message.content.strip()

# ── セッションステート ────────────────────────────
if "chat_log" not in st.session_state: st.session_state.chat_log=[]
if "summary"   not in st.session_state: st.session_state.summary=""

# ── サイドバー設定 ────────────────────────────────
st.sidebar.title("⚙️ 表示設定")
size_choice=st.sidebar.selectbox("本文サイズ",["小","中","大"])
size_map={"小":"14px","中":"18px","大":"24px"}
img_map={"小":60,"中":80,"大":110}
inject_custom_css(size_map[size_choice])

# ── ヘッダー ────────────────────────────────────
def img_b64(path): return base64.b64encode(open(path,"rb").read()).decode()
header_img=img_b64("LRADimg.png")
st.markdown(f"""
<div style="display:flex;align-items:center">
  <img src="data:image/png;base64,{header_img}" width="{img_map[size_choice]}" style="margin-right:10px;">
  <h1 style="margin:0;font-size:40px;font-weight:bold;">LRADサポートチャット</h1>
</div>""", unsafe_allow_html=True)
st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

# ── よくある質問ランダム表示 ─────────────────────
st.markdown("### 💡 よくある質問（ランダム表示）")
for _,r in common_faq_df.sample(3).iterrows():
    st.markdown(f'<div class="chat-text"><b>❓ {r.質問}</b><br>🅰️ {r.回答}</div><hr>',unsafe_allow_html=True)

st.divider()

# ── 入力フォーム ─────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    user_q=st.text_input("質問をどうぞ：")
    send=st.form_submit_button("送信")

# ╭──────────────────────────────────────────────╮
# │ メモリ方式: 要約＋直近ログ （6ターンで要約） │
# ╰──────────────────────────────────────────────╯
def update_memory():
    """6ターン超えたら古い4ターンを要約し summary に追記"""
    if len(st.session_state.chat_log) <= 6: return
    older = st.session_state.chat_log[2:]  # 古い分（先頭=最新）
    # 要約用テキスト
    history_text = "\n".join([f"ユーザー:{q}\nAI:{a}" for q,a in older[::-1]])
    summary_prompt = [
        {"role":"system","content":"以下の会話を日本語で100字以内に要約してください。"},
        {"role":"user","content":history_text}
    ]
    summary = gpt_chat(summary_prompt)
    st.session_state.summary += "\n" + summary
    # 古いログを捨てて最新2ターンだけ残す
    st.session_state.chat_log = st.session_state.chat_log[:2]

if send and user_q:
    if not is_valid_input(user_q):
        st.warning("入力が不正です。3〜300文字、記号率30%未満にしてください。")
    else:
        faq_q, faq_a = find_top_similar(user_q)
        # 会話コンテキストを組み立て
        sys_msg = {"role":"system","content":
            "あなたはLRAD（遠赤外線電子熱分解装置）の専門家です。丁寧に答えてください。"}
        mem_msg = {"role":"system","content":f"これまでの会話要約:\n{st.session_state.summary}"} if st.session_state.summary else None
        history_msgs = []
        for q,a in st.session_state.chat_log[:2][::-1]:  # 直近2ターンのみ
            history_msgs.extend([{"role":"user","content":q},{"role":"assistant","content":a}])
        user_context = {"role":"user","content":
            f"ユーザー質問:{user_q}\n参考FAQ質問:{faq_q}\n参考FAQ回答:{faq_a}"}

        msgs = [sys_msg] + ([mem_msg] if mem_msg else []) + history_msgs + [user_context]
        answer = gpt_chat(msgs)

        st.session_state.chat_log.insert(0,(user_q,answer))
        update_memory()
        st.experimental_rerun()

# ── チャット履歴表示 ─────────────────────────────
if st.session_state.chat_log:
    st.subheader("📜 チャット履歴（最新→古い）")
    for q,a in st.session_state.chat_log:
        st.markdown(f'<div class="chat-text"><b>🧑‍💻 質問:</b> {q}<br><b>🤖 回答:</b> {a}</div><hr>',unsafe_allow_html=True)
