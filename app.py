import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, random, re, unicodedata
import base64

# ページ設定
st.set_page_config(page_title="LRADサポートチャット", layout="centered")

# OpenAIキー
openai.api_key = st.secrets["OpenAIAPI"]["openai_api_key"]

# CSS注入（文字サイズとラベル・キャプション対応）
def inject_custom_css(body_font_size: str = "16px", title_font_size: str = "24px"):
    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            font-size: {body_font_size} !important;
        }}

        div[data-testid="stMarkdownContainer"] h1 {{
            font-size: {title_font_size} !important;
            line-height: 1.4;
        }}

        div[data-testid="stMarkdownContainer"] h2 {{
            font-size: calc({title_font_size} * 0.8) !important;
        }}
        div[data-testid="stMarkdownContainer"] h3 {{
            font-size: calc({title_font_size} * 0.7) !important;
        }}

        p > small {{
            font-size: calc({body_font_size} * 0.9) !important;
        }}

        div[data-testid="text-input-label"] > div,
        input[type="text"],
        input[type="text"]::placeholder,
        button[kind], span, label {{
            font-size: {body_font_size} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# サイドバー：文字サイズ選択とCSS注入
st.sidebar.title("⚙️ 表示設定")
font_size = st.sidebar.selectbox("文字サイズを選んでください", ["小", "中", "大"])
font_size_map = {"小": "14px", "中": "18px", "大": "22px"}
img_width_map = {"小": 60, "中": 80, "大": 110}  

selected_body = font_size_map[font_size]
selected_title = str(int(selected_body.replace("px", "")) * 1.6) + "px"
selected_img = img_width_map[font_size]

inject_custom_css(body_font_size=selected_body, title_font_size=selected_title)

# アプリ名とロゴ
st.image("logo.png", width=selected_img)
st.title("LRADサポートチャット")

# 残りのアプリの本体処理（CSV読込、ベクトル化、類似検索、OpenAI回答など）
# ...
