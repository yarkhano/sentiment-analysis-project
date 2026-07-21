"""
Sentiment Analysis - Interactive Demo UI
Run with: streamlit run app.py
"""

import re
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ---------- Page config ----------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextArea textarea {
        font-size: 1.05rem;
        border-radius: 12px;
        border: 1px solid #2b2f3a;
    }
    .result-card {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.4s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .positive { background: linear-gradient(135deg, #0f5132, #146c43); }
    .negative { background: linear-gradient(135deg, #58151c, #842029); }
    .neutral  { background: linear-gradient(135deg, #4a4a1f, #6b6b2b); }
    .result-label { font-size: 2rem; font-weight: 700; color: white; }
    .result-sub { font-size: 1rem; color: #d0d0d0; margin-top: 0.25rem; }
    .history-item {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        background-color: #1a1c24;
        border-left: 4px solid #444;
    }
    </style>
""", unsafe_allow_html=True)


# ---------- Model loading (cached so it only loads once) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


@st.cache_resource
def get_lemmatizer_and_stopwords():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")) - {"not", "no", "very"}
    return lemmatizer, stop_words


def clean_text(text, lemmatizer, stop_words):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJI = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}
CSS_CLASS = {"Negative": "negative", "Neutral": "neutral", "Positive": "positive"}
COLORS = {"Negative": "#dc3545", "Neutral": "#c9a900", "Positive": "#28a745"}


# ---------- Header ----------
st.markdown("<h1 style='text-align:center;'>💬 Product Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#9aa0a6;'>TF-IDF + Logistic Regression — type a review below to see the predicted sentiment in real time.</p>",
    unsafe_allow_html=True,
)
st.write("")

if "history" not in st.session_state:
    st.session_state.history = []

try:
    model, vectorizer = load_artifacts()
    lemmatizer, stop_words = get_lemmatizer_and_stopwords()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False
    st.error(
        "Model files not found. Make sure `sentiment_model.pkl` and `tfidf_vectorizer.pkl` "
        "are in the same folder as this app."
    )

# ---------- Input ----------
review_text = st.text_area(
    "Write a product review:",
    placeholder="e.g. The battery life is amazing but the packaging was damaged...",
    height=120,
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
with col2:
    clear_clicked = st.button("🗑️ Clear History", use_container_width=True)

if clear_clicked:
    st.session_state.history = []
    st.rerun()

# ---------- Prediction ----------
if analyze_clicked and artifacts_loaded:
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(review_text, lemmatizer, stop_words)
        vec = vectorizer.transform([cleaned])
        pred_class = model.predict(vec)[0]
        pred_label = LABELS[pred_class]

        # confidence via decision_function or predict_proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
        else:
            scores = model.decision_function(vec)[0]
            exp_scores = pd.Series(scores).apply(lambda x: max(x, 0) + 1e-6)
            probs = (exp_scores / exp_scores.sum()).values

        confidence = max(probs) * 100

        st.markdown(
            f"""
            <div class="result-card {CSS_CLASS[pred_label]}">
                <div class="result-label">{EMOJI[pred_label]} {pred_label}</div>
                <div class="result-sub">Confidence: {confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability bar chart
        fig = go.Figure(
            go.Bar(
                x=[LABELS[i] for i in range(3)],
                y=[probs[i] * 100 for i in range(3)],
                marker_color=[COLORS[LABELS[i]] for i in range(3)],
                text=[f"{probs[i]*100:.1f}%" for i in range(3)],
                textposition="outside",
            )
        )
        fig.update_layout(
            yaxis_title="Confidence (%)",
            yaxis_range=[0, 105],
            height=320,
            margin=dict(t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.session_state.history.insert(0, {"text": review_text, "label": pred_label, "confidence": confidence})

# ---------- History ----------
if st.session_state.history:
    st.markdown("### Recent Reviews")
    for item in st.session_state.history[:8]:
        st.markdown(
            f"""
            <div class="history-item" style="border-left-color:{COLORS[item['label']]};">
                <b>{EMOJI[item['label']]} {item['label']}</b> ({item['confidence']:.1f}%)
                <br><span style="color:#c0c0c0;">{item['text']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )