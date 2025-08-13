from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def batch_predict(texts):
    """
    Returns sentiment predictions for a list of texts.
    Each result includes 'label' and 'score'.
    """
    return sentiment_pipeline(texts)

# Load emotion classification pipeline
emotion_pipeline = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=False
)

def batch_emotions(texts):
    results = emotion_pipeline(texts)
    return [r["label"] for r in results]


# Emoji mapping for emotions
EMOJI_MAP = {
    "joy": "😄",
    "anger": "😠",
    "sadness": "😢",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️",
    "neutral": "😐"
}
