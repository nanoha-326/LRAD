import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# ページ設定
st.set_page_config(page_title="LRADチャット インサイト分析", layout="wide")

st.title("📊 LRADサポートチャット インサイトダッシュボード")

# ログ読み込み
LOG_FILE = "chat_logs.csv"

if not os.path.exists(LOG_FILE):
    st.warning("⚠️ チャットログ（chat_logs.csv）が見つかりません。チャット送信後に自動保存されます。")
    st.stop()

df = pd.read_csv(LOG_FILE)

if df.empty:
    st.info("チャットログがまだ保存されていません。")
    st.stop()

# タイムスタンプ処理
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

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

st.sidebar.write(f"表示件数: {len(filtered_df)}件")

# グラフ1：よくある質問ランキング
st.subheader("📌 よくある質問ランキング（Top 10）")
top_questions = filtered_df["question"].value_counts().head(10)

st.bar_chart(top_questions)

# グラフ2：時間帯別の質問傾向
st.subheader("🕒 時間帯別の質問数（0〜23時）")
hourly_counts = filtered_df.groupby("hour").size().reindex(range(24), fill_value=0)

fig1, ax1 = plt.subplots()
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=ax1, palette="Blues_d")
ax1.set_xlabel("時間帯")
ax1.set_ylabel("質問数")
st.pyplot(fig1)

# グラフ3：日別の質問推移
st.subheader("📅 日別の質問数")
daily_counts = filtered_df.groupby("date").size()

st.line_chart(daily_counts)

# オプション：最近の質問一覧
with st.expander("🗂 最近の質問一覧を表示", expanded=False):
    st.dataframe(
        filtered_df[["timestamp", "question", "answer"]].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# FAQヒット率（参考用：is_faq_matched カラムがある場合）
if "faq_matched" in df.columns:
    st.subheader("✅ FAQヒット率")
    match_rate = df["faq_matched"].mean()
    st.metric(label="FAQヒット率", value=f"{match_rate*100:.1f} %")

st.caption("※ この分析は `chat_logs.csv` に記録されたチャットログに基づいています。")
