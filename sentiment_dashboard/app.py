import os
from pathlib import Path
import io

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from wordcloud import WordCloud
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────────────────────────────────────
# Page config & Branding
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sentiment & Emotion Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cache-heavy resources (pipelines)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

@st.cache_resource(show_spinner=False)
def load_emotion_pipeline():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False,
    )

sentiment_pipe = load_sentiment_pipeline()
emotion_pipe  = load_emotion_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# Emoji map
# ─────────────────────────────────────────────────────────────────────────────
EMOJI_MAP = {
    "joy":     "😄",
    "sadness": "😢",
    "anger":   "😠",
    "fear":    "😨",
    "love":    "❤️",
    "surprise":"😲",
    "neutral": "😐",
    "disgust": "🤢",
    "shame":   "😳",
    "guilt":   "😔",
    "confusion": "😕",
    "admiration": "😍",
    "excitement": "🤩",
    "pride": "😌",
    "relief": "😌",
    "embarrassment": "😳",
    "boredom": "😒",
    "curiosity": "🤔",
    "nervousness": "😰",
    "anticipation": "🤔",
    "gratitude": "🙏",
    "hope": "🌟",
    "disappointment": "😞",
    "frustration": "😤",
    "contentment": "😊",
    "confident": "😎",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def batch_predict(texts):
    return sentiment_pipe(texts, truncation=True)

def batch_emotions(texts):
    results = emotion_pipe(texts, truncation=True)
    # handle both list-of-lists and flat lists
    if isinstance(results[0], list):
        return [r[0]["label"] for r in results]
    return [r["label"] for r in results]

@st.cache_data(show_spinner=False)
def analyze_file(uploaded_file_bytes, threshold):
    uploaded_file_bytes.seek(0)
    name = getattr(uploaded_file_bytes, "name", "")
    ext  = Path(name).suffix.lower()

    # load into DataFrame
    try:
        if ext == ".csv":
            df = pd.read_csv(uploaded_file_bytes)
        elif ext in (".xls", ".xlsx"):
            eng = "xlrd" if ext == ".xls" else "openpyxl"
            df = pd.read_excel(uploaded_file_bytes, engine=eng)
        else:
            try:
                df = pd.read_csv(uploaded_file_bytes)
            except:
                uploaded_file_bytes.seek(0)
                df = pd.read_excel(uploaded_file_bytes, engine="openpyxl")
    except ValueError as ve:
        if "specify an engine" in str(ve):
            uploaded_file_bytes.seek(0)
            df = pd.read_excel(uploaded_file_bytes, engine="openpyxl")
        else:
            raise ve

    # find text column
    cols_map = {c.lower(): c for c in df.columns}
    if "text" in cols_map:
        tc = cols_map["text"]
    elif len(df.columns)==1:
        tc = df.columns[0]
    else:
        raise ValueError(f"Missing a `text` column. Found: {list(df.columns)}")
    if tc!="text":
        df = df.rename(columns={tc:"text"})

    texts      = df["text"].astype(str).tolist()
    sentiments = batch_predict(texts)
    emotions   = batch_emotions(texts)

    df["sentiment"] = [
        s["label"] if s["score"]>=threshold else "NEUTRAL"
        for s in sentiments
    ]
    df["score"]      = [s["score"] for s in sentiments]
    df["emotion"]    = emotions
    df["emotion_icon"]= df["emotion"].str.lower().map(EMOJI_MAP).fillna("❓")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: settings & legend
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Analysis Settings")
sentiment_threshold = st.sidebar.slider(
    "Sentiment score threshold",
    min_value=0.0, max_value=1.0,
    value=0.6, step=0.05,
    help="Above this → POS/NEG; otherwise NEUTRAL"
)
with st.sidebar.expander("Advanced"):
    batch_size = st.number_input(
        "Batch size", min_value=1, max_value=100, value=32
    )
with st.sidebar.expander("Emoji Legend"):
    for emo, icon in EMOJI_MAP.items():
        st.markdown(f"- **{emo.capitalize()}**: {icon}")

# ─────────────────────────────────────────────────────────────────────────────
# Main UI – tabs
# ─────────────────────────────────────────────────────────────────────────────
st.title("AI Sentiment & Emotion Analyzer")
st.markdown("---")

tab_upload, tab_manual, tab_results, tab_history = st.tabs([
    "📤 Upload Data", "✍️ Manual Input", "📊 Results", "📜 History"
])

# ─────────────────────────────────────────────────────────────────────────────
# Upload Data
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("⚙️ How It Works")
    st.markdown(
        "1. Upload a CSV/XLS(X) with a `text` column  \n"
        "2. Adjust threshold in sidebar  \n"
        "3. Explore & export results"
    )
    st.markdown("[📄 Download sample template](https://example.com/sample_template.csv)")
    file = st.file_uploader(
        "Drag & drop your file", type=["csv","xls","xlsx"],
        help="Max 200MB"
    )
    if file:
        try:
            df = analyze_file(file, sentiment_threshold)
        except Exception as e:
            st.error(str(e))
        else:
            st.success("File processed!")
            st.session_state.history.append({
                "type":"file",
                "time":pd.Timestamp.now(),
                "source":getattr(file,"name","uploaded"),
                "entries":len(df)
            })
            with st.expander("Preview first 5 rows"):
                st.dataframe(df.head(), use_container_width=True)
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download enriched CSV",
                data=csv_out,
                file_name="analyzed_results.csv",
                mime="text/csv"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Manual Input
# ─────────────────────────────────────────────────────────────────────────────
with tab_manual:
    st.subheader("✏️ On-the-Fly Analysis")
    user_txt = st.text_area("Enter text here", height=200, placeholder="I love this product!")
    if st.button("Analyze"):
        if not user_txt.strip():
            st.warning("Please type something first.")
        else:
            sent       = batch_predict([user_txt])[0]
            emo_result = batch_emotions([user_txt])
            emo        = emo_result[0] if emo_result else "unknown"
            score      = sent["score"]
            label      = sent["label"] if score>=sentiment_threshold else "NEUTRAL"
            emoji      = EMOJI_MAP.get(emo.lower(),"❓")

            st.metric("Sentiment", label, delta=f"{score:.2f}")
            st.metric("Emotion", f"{emo} {emoji}")

            if emoji=="❓":
                st.info(f"Emotion '{emo}' not in map.")
                st.json(emo_result)

            st.session_state.history.append({
                "type":"manual","time":pd.Timestamp.now(),
                "text":user_txt,"sent":label,
                "score":score,"emotion":emo
            })

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
with tab_results:
    st.subheader("📈 Visual Insights")
    if "df" in locals() and df is not None:
        # PX bar charts (fixed column names)
        sent_counts = df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment","count"]
        fig_s = px.bar(
            sent_counts, x="sentiment", y="count",
            title="Sentiment Distribution"
        )

        emo_counts = df["emotion"].value_counts().reset_index()
        emo_counts.columns = ["emotion","count"]
        fig_e = px.bar(
            emo_counts, x="emotion", y="count",
            title="Emotion Distribution"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_s, use_container_width=True)
        with c2:
            st.plotly_chart(fig_e, use_container_width=True)

        # Donut chart (Altair)
        st.markdown("#### Emotion Breakdown (Donut)")
        donut_df = (
            df["emotion"]
            .value_counts()
            .rename_axis("emotion")
            .reset_index(name="count")
        )
        donut = (
            alt.Chart(donut_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("emotion:N"),
                tooltip=["emotion","count"]
            )
        )
        st.altair_chart(donut, use_container_width=True)

        # Advanced toggle
        show_adv = st.checkbox("Show advanced charts", value=True)
        if show_adv:
            # Word clouds grid
            st.markdown("### Word Clouds by Emotion")
            emotions = df["emotion"].unique().tolist()
            cols_per_row = 3
            cols = st.columns(cols_per_row)
            for idx, emo in enumerate(emotions):
                col = cols[idx % cols_per_row]
                texts = " ".join(df.loc[df["emotion"]==emo, "text"])
                wc = WordCloud(width=400, height=200, background_color="white").generate(texts)
                col.image(wc.to_array(), caption=f"{emo} {EMOJI_MAP.get(emo,'')}", use_column_width=True)

            # Full results table
            st.markdown("### Full Results Table")
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Upload data or enter text to see results here.")

# ─────────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("📜 Analysis History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export History as CSV",
            data=csv_hist,
            file_name="analysis_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history.clear()
            st.experimental_rerun()
    else:
        st.info("No history yet. Run some analyses!")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; padding: 10px 0; font-size: 0.9em;">
        © 2025 LEYA • Built with 🤖 AI & ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
