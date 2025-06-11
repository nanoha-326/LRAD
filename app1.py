import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(
    st.secrets["GoogleSheets"]["service_account_info"],
    scopes=scope
)

gc = gspread.authorize(creds)
sh = gc.open_by_key(st.secrets["GoogleSheets"]["sheet_key"])  # ←ここで止まってた
worksheet = sh.sheet1

worksheet.update("A1", "✅ 書き込み成功テスト")
st.success("スプレッドシートに書き込み成功！")
