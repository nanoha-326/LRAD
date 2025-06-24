import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import gspread
import json
import numpy as np
from google.oauth2.service_account import Credentials
from openai import OpenAI
from sklearn.cluster import KMeans


CORRECT_PASSWORD = "123"

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

if not st.session_state["is_admin"]:
    with st.form("admin_login_form"):
        st.title("🔐 管理者専用 Insights ダッシュボード")
        password = st.text_input("パスワードを入力してください", type="password")
        submitted = st.form_submit_button("ログイン")

        if submitted:
            if password == CORRECT_PASSWORD:
                st.session_state["is_admin"] = True
                st.success("ログイン成功")
                st.experimental_rerun()
            else:
                st.error("パスワードが間違っています。")
    st.stop()
    
# ページ設定（早めに）
st.set_page_config(page_title="LRADチャット インサイト分析", layout="wide")
st.title("📊 LRADサポートチャット インサイトダッシュボード")

# OpenAIクライアント初期化
client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)

# 質問テキストリストからEmbeddingを取得する関数
def get_embeddings(texts):
    embeddings = []
    batch_size = 20  # API制限を考慮
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Timestampやdate型を文字列に変換する関数（Google Sheets保存用）
def convert_timestamps_to_str(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            if not df[col].empty and isinstance(df[col].iloc[0], datetime.date):
                df[col] = df[col].astype(str)
    return df

# Google Sheetsに保存する関数
def save_insight_to_gsheet(data: pd.DataFrame, sheet_name: str):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    raw_info = st.secrets["GoogleSheets"]["service_account_info"]
    if isinstance(raw_info, str):
        info = json.loads(raw_info)
    else:
        info = raw_info
    creds = Credentials.from_service_account_info(info, scopes=scope)
    gc = gspread.authorize(creds)
    sheet_key = st.secrets["GoogleSheets"]["sheet_key"]
    sh = gc.open_by_key(sheet_key)
    try:
        worksheet = sh.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="20")
    worksheet.clear()
    data_to_save = convert_timestamps_to_str(data.copy())
    worksheet.update([data_to_save.columns.values.tolist()] + data_to_save.values.tolist())

# ログ読み込み
LOG_FILE = "chat_logs.csv"
if not os.path.exists(LOG_FILE):
    st.warning("⚠️ チャットログ（chat_logs.csv）が見つかりません。")
    st.stop()

df = pd.read_csv(LOG_FILE)

if df.empty:
    st.info("チャットログがまだ保存されていません。")
    st.stop()

# タイムスタンプ整形
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.to_period("M").astype(str)

# サイドバー：日付絞り込み
st.sidebar.header("🔍 絞り込み")
min_date = df["date"].min()
max_date = df["date"].max()
selected_range = st.sidebar.date_input("表示する期間", (min_date, max_date))

if isinstance(selected_range, tuple):
    start_date, end_date = selected_range
else:
    start_date = end_date = selected_range

filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
st.sidebar.write(f"表示件数: {len(filtered_df)} 件")

# 質問テキストのEmbedding取得＆クラスタリング
questions = filtered_df["question"].fillna("").tolist()

if len(questions) > 0:
    with st.spinner("質問をベクトル化中..."):
        embeddings = get_embeddings(questions)

    if embeddings.shape[0] < 2:
        st.warning("質問が少なすぎてクラスタリングをスキップします。")
    else:
        # NaNチェック
        if np.isnan(embeddings).any():
            st.error("Embeddingに無効な値が含まれています。")
        else:
            # クラスタ数は質問数未満に調整
            num_clusters = min(5, embeddings.shape[0])
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            filtered_df["cluster"] = clusters

            st.subheader("質問の自動クラスタリング結果")
            for cluster_num in range(num_clusters):
                st.write(f"### クラスタ {cluster_num + 1}")
                cluster_questions = filtered_df[filtered_df["cluster"] == cluster_num]["question"]
                if not cluster_questions.empty:
                    st.write(f"代表質問例: {cluster_questions.iloc[0]}")
                    st.write(f"質問数: {len(cluster_questions)}")
                    with st.expander("質問一覧を表示"):
                        st.write(cluster_questions.tolist())


# Google Sheets保存ボタン
if st.button("📤 Google Sheetsに保存（Insights）"):
    try:
        save_insight_to_gsheet(filtered_df, sheet_name="Insights")
        st.success("✅ Google Sheets に保存しました！")
    except Exception as e:
        st.error(f"❌ 保存に失敗しました: {e}")

# グラフ1：よくある質問ランキング
st.subheader("📌 よくある質問ランキング（Top 10）")
top_questions = filtered_df["question"].value_counts().head(10)
st.bar_chart(top_questions)

# グラフ2：時間帯別の質問数
st.subheader("🕒 時間帯別の質問数（0〜23時）")
hourly_counts = filtered_df.groupby("hour").size().reindex(range(24), fill_value=0)
fig1, ax1 = plt.subplots()
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=ax1, palette="Blues_d")
ax1.set_xlabel("時間帯")
ax1.set_ylabel("質問数")
st.pyplot(fig1)

# グラフ3：月別の質問数
st.subheader("🗓 月別の質問数")
monthly_counts = filtered_df.groupby("month").size()
st.line_chart(monthly_counts)

# グラフ4：カテゴリ別の質問数
if "category" in df.columns:
    st.subheader("🏷 カテゴリ別の質問数")
    category_counts = filtered_df["category"].value_counts()
    st.bar_chart(category_counts)

# グラフ5：FAQ外の質問割合
if "faq_matched" in df.columns:
    st.subheader("❓ FAQ外質問の割合")
    matched_counts = filtered_df["faq_matched"].value_counts(normalize=True) * 100
    labels = ["FAQに該当", "該当せず"]
    values = [matched_counts.get(True, 0), matched_counts.get(False, 0)]
    fig2, ax2 = plt.subplots()
    ax2.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#66b3ff", "#ff9999"])
    ax2.axis("equal")
    st.pyplot(fig2)

# 最近の質問一覧
with st.expander("🗂 最近の質問一覧を表示", expanded=False):
    st.dataframe(
        filtered_df[["timestamp", "question", "answer"]].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True
    )

st.caption("※ この分析は `chat_logs.csv` に記録されたチャットログに基づいています。")
