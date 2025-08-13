import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment & Emotion Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for better styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .positive-sentiment {
        color: #16a34a;
        font-weight: bold;
    }
    .negative-sentiment {
        color: #dc2626;
        font-weight: bold;
    }
    .neutral-sentiment {
        color: #ea580c;
        font-weight: bold;
    }
    .emotion-display {
        font-size: 1.2rem;
        padding: 0.5rem;
        background-color: #f1f5f9;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Emoji map
# ─────────────────────────────────────────────────────────────────────────────
EMOJI_MAP = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
    "neutral": "😐",
    "disgust": "🤢",
    "shame": "😳",
    "guilt": "😔",
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
# Enhanced analyzer functions
# ─────────────────────────────────────────────────────────────────────────────


def run_sentiment_emotion_analysis(text, model_name):
    """
    Enhanced analyzer with more realistic outputs based on text content.
    In production, replace with actual NLP models.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "neutral", 0.5, "neutral", 0.5

    text_lower = text.lower()

    # Simple keyword-based sentiment analysis (replace with real model)
    positive_words = [
        "love",
        "great",
        "amazing",
        "wonderful",
        "fantastic",
        "excellent",
        "good",
        "happy",
        "joy",
    ]
    negative_words = [
        "hate",
        "terrible",
        "awful",
        "bad",
        "horrible",
        "sad",
        "angry",
        "frustrated",
        "disappointed",
    ]

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        sentiment = "positive"
        sent_score = min(0.6 + (pos_count * 0.15), 0.95)
        emotion = np.random.choice(["joy", "love", "excitement", "contentment"])
    elif neg_count > pos_count:
        sentiment = "negative"
        sent_score = max(0.4 - (neg_count * 0.15), 0.05)
        emotion = np.random.choice(
            ["sadness", "anger", "frustration", "disappointment"]
        )
    else:
        sentiment = "neutral"
        sent_score = 0.45 + np.random.random() * 0.1
        emotion = np.random.choice(["neutral", "curiosity", "confusion"])

    emotion_score = min(0.5 + np.random.random() * 0.4, 0.95)

    return sentiment, round(sent_score, 2), emotion, round(emotion_score, 2)


def load_sample_data():
    """Return a more comprehensive sample DataFrame."""
    return pd.DataFrame(
        {
            "text": [
                "I absolutely love this product! It's amazing!",
                "This is so frustrating and disappointing...",
                "What a wonderful surprise to see you here!",
                "I'm feeling quite neutral about this situation.",
                "This makes me incredibly angry and upset!",
                "I'm so grateful for all your help and support.",
                "This is boring and doesn't interest me at all.",
                "I'm curious to know more about this topic.",
                "I feel ashamed about what happened yesterday.",
                "This gives me hope for a better future.",
            ]
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Initialize session state with error handling
# ─────────────────────────────────────────────────────────────────────────────


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "history": [],
        "emoji_map": EMOJI_MAP.copy(),
        "model_name": "basic",
        "theme_choice": "Light",
        "batch_results": None,
        "analysis_count": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def get_sentiment_color(sentiment, score, threshold):
    """Get color class for sentiment display."""
    if sentiment == "positive" or score >= threshold:
        return "positive-sentiment"
    elif sentiment == "negative" or score <= (1 - threshold):
        return "negative-sentiment"
    else:
        return "neutral-sentiment"


def validate_dataframe(df):
    """Validate uploaded dataframe."""
    if df is None or df.empty:
        return False, "File is empty."

    if "text" not in df.columns:
        available_cols = ", ".join(df.columns)
        return False, f"'text' column not found. Available columns: {available_cols}"

    return True, "Valid"


def process_text_safely(text):
    """Safely process text input."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Enhanced Controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎛️ Controls")

    # Session stats
    st.info(f"📊 Analyses run: {st.session_state.analysis_count}")

    # 1) Data Upload / Sample Download
    st.subheader("📁 Data Management")
    uploaded_file = st.file_uploader(
        "Upload CSV or XLSX",
        type=["csv", "xlsx"],
        help="File should contain a 'text' column",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Get Sample", use_container_width=True):
            sample_df = load_sample_data()
            st.session_state.sample_data = sample_df

    with col2:
        if "sample_data" in st.session_state:
            st.download_button(
                "💾 Download",
                data=st.session_state.sample_data.to_csv(index=False),
                file_name="sample_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()

    # 2) Quick Text Analysis
    st.subheader("✍️ Quick Analysis")
    quick_text = st.text_area(
        "Enter text here", height=100, placeholder="Type something to analyze..."
    )

    analyze_button = st.button(
        "🔍 Analyze Now", type="primary", use_container_width=True
    )

    st.divider()

    # 3) Analysis Settings
    st.subheader("⚙️ Settings")
    threshold = st.slider(
        "Sentiment Threshold",
        0.0,
        1.0,
        0.6,
        0.05,
        help="Above this = positive, below (1-threshold) = negative",
    )
    confidence = st.slider(
        "Emotion Confidence",
        0.0,
        1.0,
        0.5,
        0.05,
        help="Minimum emotion confidence to display",
    )
    model_choice = st.selectbox(
        "Select Model", ["basic", "advanced", "premium"], key="model_name"
    )

    st.divider()

    # 4) Emoji Map Customization
    with st.expander("🎨 Customize Emojis"):
        st.caption("Personalize emotion emojis")

        # Create columns for better layout
        emotions_list = list(st.session_state.emoji_map.keys())
        mid_point = len(emotions_list) // 2

        col1, col2 = st.columns(2)

        with col1:
            for emotion in emotions_list[:mid_point]:
                new_emoji = st.text_input(
                    emotion.title(),
                    st.session_state.emoji_map[emotion],
                    key=f"emoji_{emotion}",
                    max_chars=4,
                )
                st.session_state.emoji_map[emotion] = new_emoji

        with col2:
            for emotion in emotions_list[mid_point:]:
                new_emoji = st.text_input(
                    emotion.title(),
                    st.session_state.emoji_map[emotion],
                    key=f"emoji_{emotion}",
                    max_chars=4,
                )
                st.session_state.emoji_map[emotion] = new_emoji

    st.divider()

    # 5) Theme and Preferences
    st.subheader("🎨 Preferences")
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], key="theme_choice")

    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_emojis = st.checkbox("Show emojis", value=True)

    st.divider()

    # 6) Data Management
    st.subheader("🗂️ Data Management")
    if st.session_state.history:
        st.caption(f"History: {len(st.session_state.history)} items")
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history.clear()
            st.session_state.analysis_count = 0
            st.success("History cleared!")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Process quick analysis
# ─────────────────────────────────────────────────────────────────────────────
if analyze_button:
    if quick_text and quick_text.strip():
        with st.spinner("Analyzing..."):
            sentiment, s_score, emotion, e_score = run_sentiment_emotion_analysis(
                quick_text, st.session_state.model_name
            )
            st.session_state.history.append(
                {
                    "text": (
                        quick_text[:50] + "..." if len(quick_text) > 50 else quick_text
                    ),
                    "full_text": quick_text,
                    "sentiment": sentiment,
                    "sentiment_score": s_score,
                    "emotion": emotion,
                    "emotion_score": e_score,
                }
            )
            st.session_state.analysis_count += 1
        st.success("✅ Analysis complete!")
    else:
        st.error("❌ Please enter some text to analyze.")

# ─────────────────────────────────────────────────────────────────────────────
# Main Area: Enhanced Tabs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🤖 Sentiment & Emotion Analyzer</div>',
    unsafe_allow_html=True,
)

tab_home, tab_single, tab_batch, tab_viz, tab_history, tab_insights = st.tabs(
    [
        "🏠 Home",
        "🔍 Single Analysis",
        "📊 Batch Results",
        "📈 Visualizations",
        "📋 History",
        "💡 Insights",
    ]
)

# --- Enhanced Home Tab ---
with tab_home:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
        ### Welcome to the Advanced Sentiment & Emotion Analyzer! 🎉

        This powerful tool helps you analyze text for sentiment and emotions using advanced NLP techniques.

        #### 🚀 Features:
        - **Real-time Analysis**: Instant sentiment and emotion detection
        - **Batch Processing**: Upload CSV/XLSX files for bulk analysis
        - **Interactive Visualizations**: Beautiful charts and insights
        - **Customizable**: Personalize emojis and thresholds
        - **Export Ready**: Download results in multiple formats

        #### 📋 How to Use:
        1. **Quick Analysis**: Enter text in the sidebar and click "Analyze Now"
        2. **Batch Upload**: Upload a CSV/XLSX file with a 'text' column
        3. **Explore Results**: View detailed analysis across different tabs
        4. **Customize**: Adjust settings and emoji mappings to your preference

        #### 🎯 Get Started:
        - Try the sample data or enter your own text
        - Explore the different tabs for comprehensive analysis
        - Adjust settings in the sidebar for personalized results
        """
        )

        # Quick stats if there's data
        if st.session_state.history:
            st.success(
                f"🎊 Great! You've already analyzed {len(st.session_state.history)} texts. Check out the other tabs for detailed insights!"
            )

# --- Enhanced Single Analysis Tab ---
with tab_single:
    st.header("🔍 Single Text Analysis")

    if quick_text and quick_text.strip():
        with st.spinner("🔄 Analyzing your text..."):
            sentiment, s_score, emotion, e_score = run_sentiment_emotion_analysis(
                quick_text, st.session_state.model_name
            )

        # Create columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            # Sentiment Analysis
            st.subheader("😊 Sentiment Analysis")
            color_class = get_sentiment_color(sentiment, s_score, threshold)

            st.markdown(
                f"""
            <div class="metric-container">
                <h4>Sentiment: <span class="{color_class}">{sentiment.title()}</span></h4>
                <p>Score: <strong>{s_score:.2f}</strong> {'✅' if show_confidence else ''}</p>
                <div style="background: linear-gradient(90deg,
                    {'#dc2626' if s_score < 0.3 else '#ea580c' if s_score < 0.7 else '#16a34a'} {s_score*100}%,
                    #e5e7eb {s_score*100}%);
                    height: 10px; border-radius: 5px; margin: 10px 0;"></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            # Emotion Analysis
            st.subheader("🎭 Emotion Analysis")
            if e_score >= confidence:
                emoji_char = (
                    st.session_state.emoji_map.get(emotion, "❓") if show_emojis else ""
                )
                st.markdown(
                    f"""
                <div class="emotion-display">
                    <strong>{emotion.title()}</strong> {emoji_char}<br>
                    <small>Confidence: {e_score:.2f} {'✅' if show_confidence else ''}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    f"🔍 Emotion confidence ({e_score:.2f}) is below your threshold ({confidence:.2f})"
                )

        # Text preview
        st.subheader("📝 Analyzed Text")
        st.text_area("", value=quick_text, height=100, disabled=True)

        # Additional insights
        with st.expander("🔬 Detailed Analysis"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Word Count", len(quick_text.split()))
            col2.metric("Character Count", len(quick_text))
            col3.metric("Sentiment Score", f"{s_score:.3f}")
            col4.metric("Emotion Score", f"{e_score:.3f}")
    else:
        st.info(
            "👈 Enter some text in the sidebar and click 'Analyze Now' to see detailed analysis here!"
        )

        # Show example
        st.markdown(
            """
        ### 💡 Try These Examples:
        - "I absolutely love this new feature! It's amazing!"
        - "This is really frustrating and disappointing..."
        - "I'm curious about how this technology works."
        """
        )

# --- Enhanced Batch Results Tab ---
with tab_batch:
    st.header("📊 Batch Analysis Results")

    if uploaded_file:
        try:
            # Show file info
            st.info(
                f"📁 Processing file: {uploaded_file.name} ({uploaded_file.size} bytes)"
            )

            # Check file size (e.g., max 50MB)
            if uploaded_file.size > 50 * 1024 * 1024:
                st.error(
                    "❌ File size exceeds 50MB limit. Please upload a smaller file."
                )
                st.stop()

            # Read file based on extension
            with st.spinner("📖 Reading file..."):
                if uploaded_file.name.endswith(".csv"):
                    try:
                        df = pd.read_csv(uploaded_file)
                    except pd.errors.ParserError:
                        st.error("❌ Invalid CSV format. Please check your file.")
                        st.stop()
                else:
                    try:
                        df = pd.read_excel(uploaded_file)
                    except ValueError:
                        st.error("❌ Invalid XLSX format. Please check your file.")
                        st.stop()

            # Validate dataframe
            is_valid, message = validate_dataframe(df)
            if not is_valid:
                st.error(f"❌ {message}")
                st.info(
                    "💡 **Tip**: Your file should have a 'text' column containing valid text data."
                )
                st.stop()

            st.success(f"✅ File loaded successfully! Found {len(df)} rows.")

            # Show preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head(), use_container_width=True)

            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                max_rows = st.number_input(
                    "Max rows to process", 1, len(df), min(len(df), 1000)
                )
            with col2:
                process_button = st.button("🚀 Start Processing", type="primary")

            if process_button or "batch_results" in st.session_state:
                # Process the data
                with st.spinner("🔄 Analyzing texts... This may take a moment."):
                    results = []
                    progress_bar = st.progress(0)

                    df_subset = df.head(max_rows)
                    for i, (_, row) in enumerate(df_subset.iterrows()):
                        text = process_text_safely(row.get("text", ""))
                        if text:
                            res = run_sentiment_emotion_analysis(
                                text, st.session_state.model_name
                            )
                        else:
                            res = ("neutral", 0.5, "neutral", 0.5)
                        results.append(res)

                        # Update progress less frequently
                        if (i + 1) % 10 == 0 or i == len(df_subset) - 1:
                            progress_bar.progress((i + 1) / len(df_subset))

                    progress_bar.empty()

                # Create results dataframe
                res_df = pd.DataFrame(
                    results,
                    columns=[
                        "sentiment",
                        "sentiment_score",
                        "emotion",
                        "emotion_score",
                    ],
                )
                out_df = pd.concat([df_subset.reset_index(drop=True), res_df], axis=1)
                st.session_state.batch_results = out_df

                st.success(f"🎉 Processing complete! Analyzed {len(results)} texts.")

                # Display results with styling
                def color_sentiment_score(val):
                    try:
                        val = float(val)
                        if val >= threshold:
                            return "background-color: #dcfce7; color: #166534"
                        elif val <= (1 - threshold):
                            return "background-color: #fef2f2; color: #991b1b"
                        else:
                            return "background-color: #fff7ed; color: #9a3412"
                    except (ValueError, TypeError):
                        return ""

                # Apply styling
                if "sentiment_score" in out_df.columns:
                    styled_df = out_df.style.applymap(
                        color_sentiment_score, subset=["sentiment_score"]
                    ).format({"sentiment_score": "{:.2f}", "emotion_score": "{:.2f}"})
                else:
                    styled_df = out_df

                st.dataframe(styled_df, use_container_width=True)

                # Summary statistics
                st.subheader("📊 Summary Statistics")
                sent_count = Counter(out_df["sentiment"])
                emo_count = Counter(out_df["emotion"])
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sentiment Counts:**")
                    for sent, cnt in sent_count.items():
                        st.write(f"{sent.title()}: {cnt}")
                with col2:
                    st.write("**Top Emotions:**")
                    for emo, cnt in emo_count.most_common(5):
                        st.write(f"{emo.title()}: {cnt}")

                # Download options
                st.subheader("💾 Download Results")
                csv = out_df.to_csv(index=False)
                st.download_button("Download CSV", csv, "batch_results.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info(
                "💡 Please ensure your file is a valid CSV or XLSX with a 'text' column."
            )
    else:
        st.info("👆 Upload a CSV or XLSX file to get started with batch analysis.")
        with st.expander("💡 Upload Tips"):
            st.markdown(
                """
            **File Requirements:**
            - Supported formats: CSV, XLSX
            - Must contain a column named 'text'
            - Maximum recommended size: 50MB
            - Text should be in readable format

            **Sample File Structure:**
            text,category
            "I love this product!",review
            "This is disappointing...",feedback
            "Amazing service!",testimonial
            """
            )

# --- Enhanced Visualizations Tab ---
with tab_viz:
    st.header("📈 Interactive Visualizations")

    # Determine data source
    viz_data = None
    data_source = ""

    if (
        "batch_results" in st.session_state
        and st.session_state.batch_results is not None
    ):
        viz_data = st.session_state.batch_results
        data_source = "batch analysis"
    elif st.session_state.history:
        viz_data = pd.DataFrame(st.session_state.history)
        data_source = "analysis history"

    if viz_data is not None and not viz_data.empty:
        # Verify required columns
        required_columns = ["sentiment", "sentiment_score", "emotion", "emotion_score"]
        missing_columns = [
            col for col in required_columns if col not in viz_data.columns
        ]
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info(
                "💡 Please ensure your data includes sentiment and emotion analysis results."
            )
            st.stop()

        st.info(f"📊 Showing visualizations from {data_source} ({len(viz_data)} items)")

        # Tabs for different visualization types
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
            ["📊 Distributions", "📈 Trends", "🎯 Detailed Analysis", "📋 Summary"]
        )

        with viz_tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("😊 Sentiment Distribution")
                sentiment_counts = viz_data["sentiment"].value_counts()
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={
                        "positive": "#16a34a",
                        "negative": "#dc2626",
                        "neutral": "#ea580c",
                    },
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)

            with col2:
                st.subheader("🎭 Emotion Distribution")
                emotion_counts = viz_data["emotion"].value_counts().head(10)
                fig_emotion = px.bar(
                    x=emotion_counts.index,
                    y=emotion_counts.values,
                    title="Top 10 Emotions",
                    color=emotion_counts.values,
                    color_continuous_scale="viridis",
                )
                fig_emotion.update_xaxes(tickangle=45)
                st.plotly_chart(fig_emotion, use_container_width=True)

        with viz_tab2:
            st.subheader("📈 Score Distributions")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Sentiment Score Distribution**")
                fig_sent_hist = px.histogram(
                    viz_data,
                    x="sentiment_score",
                    nbins=20,
                    title="Sentiment Score Distribution",
                    color_discrete_sequence=["#3b82f6"],
                )
                st.plotly_chart(fig_sent_hist, use_container_width=True)

            with col2:
                st.write("**Emotion Score Distribution**")
                fig_emo_hist = px.histogram(
                    viz_data,
                    x="emotion_score",
                    nbins=20,
                    title="Emotion Score Distribution",
                    color_discrete_sequence=["#8b5cf6"],
                )
                st.plotly_chart(fig_emo_hist, use_container_width=True)

        with viz_tab3:
            st.subheader("🎯 Detailed Analysis")

            # Scatter plot of sentiment vs emotion scores
            fig_scatter = px.scatter(
                viz_data,
                x="sentiment_score",
                y="emotion_score",
                color="sentiment",
                size="emotion_score",
                hover_data=["emotion"],
                title="Sentiment vs Emotion Score Relationship",
                color_discrete_map={
                    "positive": "#16a34a",
                    "negative": "#dc2626",
                    "neutral": "#ea580c",
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Heatmap of emotion-sentiment combinations
            if len(viz_data) > 5:
                st.subheader("🔥 Emotion-Sentiment Heatmap")
                heatmap_data = (
                    viz_data.groupby(["sentiment", "emotion"])
                    .size()
                    .reset_index(name="count")
                )
                pivot_data = heatmap_data.pivot(
                    index="sentiment", columns="emotion", values="count"
                ).fillna(0)

                fig_heatmap = px.imshow(
                    pivot_data,
                    title="Emotion-Sentiment Combination Frequency",
                    color_continuous_scale="blues",
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

        with viz_tab4:
            st.subheader("📋 Statistical Summary")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Analyses", len(viz_data))
            col2.metric("Avg Sentiment", f"{viz_data['sentiment_score'].mean():.2f}")
            col3.metric("Avg Emotion", f"{viz_data['emotion_score'].mean():.2f}")
            col4.metric(
                "Most Common",
                (
                    viz_data["emotion"].mode()[0]
                    if not viz_data["emotion"].mode().empty
                    else "N/A"
                ),
            )

            # Detailed statistics table
            st.subheader("📊 Detailed Statistics")
            stats_data = {
                "Metric": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                "Sentiment Score": [
                    len(viz_data),
                    round(viz_data["sentiment_score"].mean(), 3),
                    round(viz_data["sentiment_score"].std(), 3),
                    round(viz_data["sentiment_score"].min(), 3),
                    round(viz_data["sentiment_score"].quantile(0.25), 3),
                    round(viz_data["sentiment_score"].median(), 3),
                    round(viz_data["sentiment_score"].quantile(0.75), 3),
                    round(viz_data["sentiment_score"].max(), 3),
                ],
                "Emotion Score": [
                    len(viz_data),
                    round(viz_data["emotion_score"].mean(), 3),
                    round(viz_data["emotion_score"].std(), 3),
                    round(viz_data["emotion_score"].min(), 3),
                    round(viz_data["emotion_score"].quantile(0.25), 3),
                    round(viz_data["emotion_score"].median(), 3),
                    round(viz_data["emotion_score"].quantile(0.75), 3),
                    round(viz_data["emotion_score"].max(), 3),
                ],
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

            # Top emotions and sentiments
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔝 Top Emotions")
                top_emotions = viz_data["emotion"].value_counts().head(5)
                for emotion, count in top_emotions.items():
                    emoji = st.session_state.emoji_map.get(emotion, "❓")
                    percentage = (count / len(viz_data)) * 100
                    st.write(
                        f"{emoji} **{emotion.title()}**: {count} ({percentage:.1f}%)"
                    )

            with col2:
                st.subheader("📈 Sentiment Breakdown")
                sent_breakdown = viz_data["sentiment"].value_counts()
                for sentiment, count in sent_breakdown.items():
                    percentage = (count / len(viz_data)) * 100
                    color = (
                        "🟢"
                        if sentiment == "positive"
                        else "🔴" if sentiment == "negative" else "🟡"
                    )
                    st.write(
                        f"{color} **{sentiment.title()}**: {count} ({percentage:.1f}%)"
                    )

    else:
        st.info("📊 No data available for visualization yet!")
        st.markdown(
            """
### 💡 To see visualizations:
1. **Quick Analysis**: Use the sidebar to analyze some text
2. **Batch Upload**: Upload a CSV/XLSX file for comprehensive charts
3. **Build History**: Analyze multiple texts to see trends

Once you have data, you'll see:
- 📊 Interactive pie charts and bar graphs
- 📈 Distribution histograms
- 🎯 Scatter plots showing relationships
- 🔥 Heatmaps of emotion-sentiment combinations
- 📋 Detailed statistical summaries
"""
        )

# --- Enhanced History Tab ---
with tab_history:
    st.header("📋 Analysis History")

    if st.session_state.history:
        st.success(
            f"📊 You have {len(st.session_state.history)} analyses in your history"
        )

        # History controls
        col1, col2, col3 = st.columns(3)
        with col1:
            show_full_text = st.checkbox("Show full text", value=False)
        with col2:
            filter_sentiment = st.selectbox(
                "Filter by sentiment", ["All", "positive", "negative", "neutral"]
            )
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Recent first", "Oldest first", "Highest score", "Lowest score"],
            )

        # Prepare history dataframe
        hist_df = pd.DataFrame(st.session_state.history)

        # Apply filters
        if filter_sentiment != "All":
            hist_df = hist_df[hist_df["sentiment"] == filter_sentiment]

        # Apply sorting
        if sort_by == "Recent first":
            hist_df = hist_df.iloc[::-1]  # Reverse order
        elif sort_by == "Oldest first":
            pass  # Keep original order
        elif sort_by == "Highest score":
            hist_df = hist_df.sort_values("sentiment_score", ascending=False)
        elif sort_by == "Lowest score":
            hist_df = hist_df.sort_values("sentiment_score", ascending=True)

        # Display history
        if not hist_df.empty:
            # Create display dataframe
            display_df = hist_df.copy()
            if not show_full_text and "full_text" in display_df.columns:
                display_df = display_df.drop("full_text", axis=1)

            # Add emoji column
            if show_emojis:
                display_df["emoji"] = display_df["emotion"].map(
                    st.session_state.emoji_map
                )

            # Format scores
            display_df["sentiment_score"] = display_df["sentiment_score"].round(2)
            display_df["emotion_score"] = display_df["emotion_score"].round(2)

            st.dataframe(display_df, use_container_width=True)

            # Quick stats
            st.subheader("📈 Quick Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Items", len(hist_df))
            col2.metric("Avg Sentiment", f"{hist_df['sentiment_score'].mean():.2f}")
            col3.metric(
                "Most Common Emotion",
                (
                    hist_df["emotion"].mode()[0]
                    if not hist_df["emotion"].mode().empty
                    else "N/A"
                ),
            )
            col4.metric(
                "Positive Ratio", f"{(hist_df['sentiment'] == 'positive').mean():.1%}"
            )

            # Export options
            st.subheader("💾 Export History")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Export as CSV"):
                    csv_data = pd.DataFrame(st.session_state.history).to_csv(
                        index=False
                    )
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name="analysis_history.csv",
                        mime="text/csv",
                    )

            with col2:
                if st.button("📋 Export Summary"):
                    summary_data = pd.DataFrame(st.session_state.history)
                    summary = f"""
Analysis History Summary
=======================
Total analyses: {len(summary_data)}
Date range: Session data

Sentiment Distribution:
{summary_data['sentiment'].value_counts().to_string()}

Top 5 Emotions:
{summary_data['emotion'].value_counts().head().to_string()}

Average Scores:
Sentiment: {summary_data['sentiment_score'].mean():.3f}
Emotion: {summary_data['emotion_score'].mean():.3f}
"""
                    st.download_button(
                        "Download Summary",
                        data=summary,
                        file_name="history_summary.txt",
                        mime="text/plain",
                    )

            with col3:
                if st.button("🗑️ Clear All History", type="secondary"):
                    if st.button("⚠️ Confirm Clear", type="primary"):
                        st.session_state.history = []
                        st.session_state.analysis_count = 0
                        st.success("✅ History cleared!")
                        # No st.rerun() needed

        else:
            st.info(f"No items match your filter criteria.")

    else:
        st.info("📝 No analysis history yet!")
        st.markdown(
            """
### 🚀 Start Building Your History:
1. Use the **Quick Analysis** in the sidebar
2. Upload files in the **Batch Results** tab
3. Your analyses will automatically appear here

### 📊 What You'll See:
- Complete log of all your analyses
- Filtering and sorting options
- Export capabilities (CSV, summary reports)
- Quick statistics and insights
"""
        )

# --- New Insights Tab ---
with tab_insights:
    st.header("💡 Advanced Insights")

    # Determine data source for insights
    insights_data = None
    data_source = ""
    if (
        "batch_results" in st.session_state
        and st.session_state.batch_results is not None
    ):
        insights_data = st.session_state.batch_results
        data_source = "batch analysis"
    elif st.session_state.history:
        insights_data = pd.DataFrame(st.session_state.history)
        data_source = "analysis history"

    if insights_data is not None and len(insights_data) > 0:
        st.info(
            f"🎯 Generating insights from {data_source} ({len(insights_data)} items)"
        )

        # Key insights
        st.subheader("🔍 Key Insights")

        # Calculate key metrics
        avg_sentiment = insights_data["sentiment_score"].mean()
        sentiment_std = insights_data["sentiment_score"].std()
        most_common_emotion = (
            insights_data["emotion"].mode()[0]
            if not insights_data["emotion"].mode().empty
            else "N/A"
        )
        positive_ratio = (insights_data["sentiment"] == "positive").mean()

        # Generate insights
        insights = []

        if avg_sentiment > 0.7:
            insights.append(
                (
                    "🌟",
                    "Highly Positive",
                    f"Your texts show overwhelmingly positive sentiment (avg: {avg_sentiment:.2f})",
                )
            )
        elif avg_sentiment < 0.3:
            insights.append(
                (
                    "⚠️",
                    "Concerning Negativity",
                    f"Many texts show negative sentiment (avg: {avg_sentiment:.2f})",
                )
            )
        else:
            insights.append(
                (
                    "⚖️",
                    "Balanced Sentiment",
                    f"Your texts show balanced sentiment (avg: {avg_sentiment:.2f})",
                )
            )

        if sentiment_std < 0.2:
            insights.append(
                (
                    "📏",
                    "Consistent Tone",
                    f"Very consistent sentiment across texts (std: {sentiment_std:.2f})",
                )
            )
        elif sentiment_std > 0.4:
            insights.append(
                (
                    "🎢",
                    "Variable Emotions",
                    f"High emotional variability in your texts (std: {sentiment_std:.2f})",
                )
            )

        if positive_ratio > 0.8:
            insights.append(
                (
                    "😊",
                    "Optimistic Content",
                    f"{positive_ratio:.1%} of your content is positive",
                )
            )
        elif positive_ratio < 0.2:
            insights.append(
                (
                    "😟",
                    "Pessimistic Trend",
                    f"Only {positive_ratio:.1%} of your content is positive",
                )
            )

        # Display insights
        for emoji, title, description in insights:
            st.markdown(
                f"""
<div class="metric-container">
    <h4>{emoji} {title}</h4>
    <p>{description}</p>
</div>
""",
                unsafe_allow_html=True,
            )

        # Detailed analysis sections
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Content Analysis")

            # Emotion diversity
            unique_emotions = insights_data["emotion"].nunique()
            total_possible = len(EMOJI_MAP)
            emotion_diversity = unique_emotions / total_possible

            st.metric(
                "Emotion Diversity",
                f"{unique_emotions}/{total_possible}",
                f"{emotion_diversity:.1%}",
            )

            # Sentiment consistency
            sentiment_consistency = 1 - sentiment_std
            st.metric(
                "Sentiment Consistency",
                f"{sentiment_consistency:.2f}",
                "Higher is more consistent",
            )

            # Most extreme examples
            if len(insights_data) > 1:
                most_positive_idx = insights_data["sentiment_score"].idxmax()
                most_negative_idx = insights_data["sentiment_score"].idxmin()

                st.write("**Most Positive Text:**")
                if "full_text" in insights_data.columns:
                    pos_text = insights_data.loc[most_positive_idx, "full_text"]
                else:
                    pos_text = insights_data.loc[most_positive_idx, "text"]
                st.text(pos_text[:100] + "..." if len(pos_text) > 100 else pos_text)
                st.caption(
                    f"Score: {insights_data.loc[most_positive_idx, 'sentiment_score']:.2f}"
                )

        with col2:
            st.subheader("🎯 Recommendations")

            recommendations = []

            if avg_sentiment < 0.4:
                recommendations.append(
                    "Consider incorporating more positive language in your content"
                )

            if sentiment_std > 0.4:
                recommendations.append(
                    "Your content shows high emotional variability - consider maintaining consistent tone"
                )

            if unique_emotions < 5:
                recommendations.append(
                    "Try expressing a wider range of emotions to engage your audience better"
                )

            if positive_ratio < 0.3:
                recommendations.append(
                    "Balance negative content with positive elements for better engagement"
                )

            if not recommendations:
                recommendations.append(
                    "Your content shows good emotional balance and consistency!"
                )

            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

        # Advanced visualizations
        st.subheader("📈 Advanced Analysis")

        # Emotion-sentiment correlation
        if len(insights_data) > 10:
            fig_advanced = px.scatter(
                insights_data,
                x="sentiment_score",
                y="emotion_score",
                color="sentiment",
                size="emotion_score",
                hover_data=["emotion"],
                title="Sentiment vs Emotion Score Analysis",
                trendline="ols",
            )
            st.plotly_chart(fig_advanced, use_container_width=True)

        # Word cloud or emotion timeline would go here in a real implementation
        st.subheader("🔮 Predictive Insights")
        st.info(
            "💡 Based on your analysis patterns, future texts are likely to be "
            + (
                "positive"
                if avg_sentiment > 0.6
                else "negative" if avg_sentiment < 0.4 else "neutral"
            )
            + f" with an average confidence of {avg_sentiment:.1%}"
        )

    else:
        st.info("💡 Generate some analyses first to see advanced insights!")
        st.markdown(
            """
### 🎯 What You'll Get:
- **Key Insights**: Automated analysis of your content patterns
- **Content Analysis**: Emotion diversity and sentiment consistency metrics  
- **Recommendations**: Personalized suggestions for improvement
- **Advanced Visualizations**: Correlation plots and trend analysis
- **Predictive Insights**: Forecasts based on your patterns

### 🚀 How to Get Started:
1. Analyze at least 5-10 texts using quick analysis or batch upload
2. Return to this tab to see detailed insights
3. Use the recommendations to improve your content strategy
"""
        )

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔧 Model Info**")
    st.caption(f"Current: {st.session_state.model_name}")
    st.caption(f"Threshold: {threshold}")

with col2:
    st.markdown("**📊 Session Stats**")
    st.caption(f"Analyses: {st.session_state.analysis_count}")
    st.caption(f"History: {len(st.session_state.history)} items")

with col3:
    st.markdown("**ℹ️ About**")
    st.caption("Built with Streamlit")
    st.caption("Powered by AI & NLP")

st.markdown(
    """
<div style="text-align: center; color: #888; padding: 20px 0; font-size: 0.9em; border-top: 1px solid #eee; margin-top: 20px;">
🤖 © 2025 LEYA • Advanced Sentiment & Emotion Analyzer • Built with ❤️ using Streamlit & AI
<br><small>⚡ Real-time Analysis • 📊 Interactive Visualizations • 🎯 Advanced Insights</small>
</div>
""",
    unsafe_allow_html=True,
)
