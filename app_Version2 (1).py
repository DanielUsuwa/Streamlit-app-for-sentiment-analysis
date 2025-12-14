#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Sentiment Analyzer (robust startup)

Behavior:
- Detects and uses the best available sentiment analyzer:
    1) NLTK VADER if nltk and vader_lexicon are present,
    2) vaderSentiment package if installed,
    3) Lightweight fallback keyword-based analyzer (always available).
- Guards optional imports (wordcloud). If wordcloud is missing the app still runs.
- Provides clear UI info about which analyzer is used and instructions to enable better analyzers.
"""
from __future__ import annotations

import io
import re
from typing import Callable, Dict, Optional

import pandas as pd
import streamlit as st

# Visualization (use Altair for charts - present on Streamlit hosts)
import altair as alt

# Optional image support
try:
    from PIL import Image
except Exception:
    Image = None  # not critical

# Optional WordCloud import - guarded
try:
    from wordcloud import WordCloud  # type: ignore
    _HAS_WORDCLOUD = True
except Exception:
    WordCloud = None  # type: ignore
    _HAS_WORDCLOUD = False

# We'll import heavier NLP packages lazily below to avoid startup crashes.

# -------------------------
# Utilities
# -------------------------
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_HTML = re.compile(r'<.*?>')
RE_NON_ALPHANUM = re.compile(r'[^A-Za-z0-9\s]')
RE_MULTI_WS = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\d+')

def simple_clean(text: str, remove_numbers: bool = True, lower: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = RE_HTML.sub(" ", text)
    text = RE_URL.sub(" ", text)
    if lower:
        text = text.lower()
    if remove_numbers:
        text = RE_NUMBERS.sub(" ", text)
    text = RE_NON_ALPHANUM.sub(" ", text)
    text = RE_MULTI_WS.sub(" ", text).strip()
    return text

# Fallback lightweight sentiment analyzer (keyword-based)
_POS_WORDS = {
    "good", "great", "excellent", "love", "nice", "fast", "happy", "recommend", "perfect", "best", "works"
}
_NEG_WORDS = {
    "bad", "poor", "terrible", "hate", "slow", "worse", "worst", "broken", "disappoint", "missing", "problem"
}

def fallback_sentiment(text: str) -> Dict[str, float]:
    """
    Return a dict with neg, neu, pos, compound keys compatible with VADER outputs.
    Simple counts of positive/negative words with a normalized compound score.
    """
    cleaned = simple_clean(text)
    tokens = cleaned.split()
    if not tokens:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    neu = max(0, len(tokens) - pos - neg)
    # compound: pos-neg normalized to [-1,1], small smoothing
    compound = (pos - neg) / (len(tokens) + 1e-6)
    # clamp
    if compound > 1:
        compound = 1.0
    if compound < -1:
        compound = -1.0
    # normalize neg/pos/neu to sum 1
    total = pos + neg + neu
    if total == 0:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return {"neg": neg / total, "neu": neu / total, "pos": pos / total, "compound": compound}

def compound_to_label(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"

# -------------------------
# Analyzer detection
# -------------------------
def detect_analyzer() -> Dict:
    """
    Attempt to provide a sentiment scoring function: score_fn(text)->dict with neg,neu,pos,compound.
    Also return a description string to show in the UI.
    """
    # 1) Try NLTK VADER
    try:
        import nltk  # type: ignore
        # ensure vader_lexicon is available; do not crash if it's not
        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError:
            # try to download quietly
            try:
                nltk.download("vader_lexicon", quiet=True)
                nltk.data.find("sentiment/vader_lexicon")
            except Exception:
                raise ImportError("vader_lexicon not available")
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        analyzer = SentimentIntensityAnalyzer()
        return {
            "score_fn": lambda t: analyzer.polarity_scores(t),
            "name": "NLTK VADER (nltk + vader_lexicon)"
        }
    except Exception:
        # 2) Try vaderSentiment package
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            analyzer = SentimentIntensityAnalyzer()
            return {
                "score_fn": lambda t: analyzer.polarity_scores(t),
                "name": "vaderSentiment package"
            }
        except Exception:
            # 3) fallback lightweight
            return {
                "score_fn": fallback_sentiment,
                "name": "Fallback keyword-based analyzer (low accuracy)"
            }

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Analyzer (robust)", layout="wide")
st.title("Sentiment Analyzer — resilient startup")

st.markdown(
    "This app will use the best available sentiment analyzer on the host. "
    "If advanced packages (nltk + vader_lexicon or vaderSentiment) are not installed, "
    "the app falls back to a simple keyword-based analyzer so it can still run."
)

with st.sidebar:
    st.header("Options")
    uploaded = st.file_uploader("Upload CSV (must contain text column)", type=["csv"])
    use_sample = st.button("Use sample data")
    text_col_input = st.text_input("Text column name", value="review_content")
    sample_n = st.number_input("Sample N (0 = all)", min_value=0, value=0, step=1)
    want_wordcloud = st.checkbox("Generate word cloud (only if available)", value=True)
    run = st.button("Analyze")

SAMPLE_CSV = """review_content
"Good product, arrived on time, works as described."
"Terrible packaging. Cable stopped working after two days."
"Average experience — nothing special, not worth extra cost."
"Excellent! Very fast delivery and works perfectly."
"Not as described. Missing parts and poor quality."
"""

def load_sample_df():
    return pd.read_csv(io.StringIO(SAMPLE_CSV))

def read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

if uploaded is None and not use_sample:
    st.info("Upload a CSV or press 'Use sample data' to start.")
    st.stop()

df = load_sample_df() if use_sample else read_csv(uploaded)
if df is None:
    st.stop()

# auto-detect text column
cols = list(df.columns)
if text_col_input and text_col_input in cols:
    text_col = text_col_input
else:
    preferred = ["review_content", "review", "text", "comment", "content"]
    text_col = next((c for c in preferred if c in cols), cols[0] if cols else None)

if text_col is None:
    st.error("No columns detected in uploaded CSV.")
    st.stop()
st.sidebar.success(f"Using text column: {text_col}")

st.subheader("Data preview")
st.dataframe(df.head(8))

if not run:
    st.info("Click 'Analyze' to run.")
    st.stop()

# Prepare analyzer
det = detect_analyzer()
score_fn: Callable[[str], Dict[str, float]] = det["score_fn"]
analyzer_name = det["name"]

st.success(f"Analyzer selected: {analyzer_name}")
if analyzer_name != "NLTK VADER (nltk + vader_lexicon)":
    st.warning(
        "For better accuracy install `nltk` and ensure `vader_lexicon` resource is available, "
        "or install `vaderSentiment` package. Then add them to requirements.txt and redeploy."
    )

# Optionally sample
if sample_n and 0 < sample_n < len(df):
    df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

# Preprocess + score
clean_texts = []
scores_list = []
n = len(df)
progress = st.progress(0)
status = st.empty()
for i, t in enumerate(df[text_col].astype(str)):
    status.text(f"Processing {i+1}/{n} ...")
    try:
        # If using heavy analyzers we can pass cleaned or raw text; keep cleaned for fallback
        cleaned = simple_clean(t, remove_numbers=True, lower=True)
    except Exception:
        cleaned = str(t)
    clean_texts.append(cleaned)
    try:
        sc = score_fn(cleaned)
        # ensure keys present
        for k in ("neg", "neu", "pos", "compound"):
            if k not in sc:
                sc.setdefault(k, 0.0)
    except Exception:
        sc = fallback_sentiment(cleaned)
    scores_list.append(sc)
    progress.progress((i + 1) / n)
progress.empty()
status.empty()

scores_df = pd.DataFrame(scores_list)
results = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
results["_clean_text"] = clean_texts
st.subheader("Results sample")
st.dataframe(results[[text_col, "_clean_text", "label" if "label" in results.columns else "compound"]].head(10))

# Add label column derived from compound
results["_label"] = results["compound"].apply(compound_to_label)

# Distribution chart
counts = results["_label"].value_counts().reset_index()
counts.columns = ["label", "count"]
if counts.empty:
    st.warning("No labels to show.")
else:
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X("label:N", sort=None),
        y=alt.Y("count:Q"),
        color=alt.Color("label:N")
    ).properties(title="Sentiment distribution")
    st.altair_chart(chart, use_container_width=True)

# Download CSV
csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

# Word cloud if available & requested
if want_wordcloud:
    if not _HAS_WORDCLOUD:
        st.info("Word cloud unavailable: 'wordcloud' package is not installed in this environment.")
    else:
        pos_text = " ".join(results.loc[results["_label"] == "positive", "_clean_text"].astype(str))
        if not pos_text.strip():
            st.warning("No positive texts to build a word cloud.")
        else:
            wc = WordCloud(width=1200, height=600, background_color="white", colormap="Blues")
            wc.generate(pos_text)
            img = wc.to_image()
            st.image(img, use_column_width=True)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download word cloud (PNG)", data=buf, file_name="positive_wordcloud.png", mime="image/png")

st.info("If you want higher-quality sentiment results, add 'nltk' (and ensure 'vader_lexicon') or 'vaderSentiment' to requirements.txt and redeploy.")