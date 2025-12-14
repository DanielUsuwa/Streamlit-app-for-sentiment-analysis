#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Sentiment Analyzer with sentiment-specific sections and WordCloud support.

Features:
- Detects best available analyzer (NLTK VADER, vaderSentiment, or fallback keyword-based).
- Shows overall distribution and separate sections/tabs for Positive / Neutral / Negative.
  Each section includes: count & percent, sample reviews, top words, downloadable filtered CSV,
  and a WordCloud for that sentiment (if `wordcloud` package is available).
- WordCloud import is guarded so the app won't crash if the package is missing.
- No mandatory heavy dependencies — app will always start using fallback analyzer.
- UX-friendly with progress indicators, clear messages, and responsive charts.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import io
import re
from collections import Counter
from typing import Callable, Dict, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# Optional WordCloud import (guarded)
try:
    from wordcloud import WordCloud  # type: ignore
    _HAS_WORDCLOUD = True
except Exception:
    WordCloud = None  # type: ignore
    _HAS_WORDCLOUD = False

# Minimal builtin stopword list (keeps app independent of NLTK)
_BASIC_STOPWORDS = {
    "the", "and", "is", "in", "it", "of", "to", "a", "this", "that", "for", "on", "with",
    "was", "are", "as", "but", "they", "i", "we", "you", "have", "has", "been", "not",
    "be", "at", "or", "an", "so", "if", "its", "by", "from"
}

st.set_page_config(page_title="Sentiment Analyzer — Detailed Sentiment Sections", layout="wide")
st.title("Sentiment Analyzer — Detailed Sentiment Sections")
st.markdown(
    "Upload a CSV with a text column (e.g. `review_content`) and analyze sentiment. "
    "This app shows per-sentiment sections (Positive / Neutral / Negative) with samples, top words, "
    "and WordClouds when available."
)

# -------------------------
# Utilities & lightweight preprocessing
# -------------------------
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_HTML = re.compile(r'<.*?>')
RE_NON_ALPHANUM = re.compile(r'[^A-Za-z0-9\s]')
RE_MULTI_WS = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\d+')
RE_EMOJI = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)


def simple_clean(text: str, remove_numbers: bool = True, lower: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = RE_HTML.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_EMOJI.sub(" ", text)
    if lower:
        text = text.lower()
    if remove_numbers:
        text = RE_NUMBERS.sub(" ", text)
    text = RE_NON_ALPHANUM.sub(" ", text)
    text = RE_MULTI_WS.sub(" ", text).strip()
    return text


# Fallback keyword-based sentiment (keeps app always runnable)
_POS_WORDS = {
    "good", "great", "excellent", "love", "nice", "fast", "happy", "recommend", "perfect", "best",
    "works", "working", "satisfied", "okay", "awesome", "amazing", "excellent", "delivered", "quick"
}
_NEG_WORDS = {
    "bad", "poor", "terrible", "hate", "slow", "worse", "worst", "broken", "disappoint",
    "missing", "problem", "return", "refund", "awful", "damage", "delay", "not"
}


def fallback_sentiment(text: str) -> Dict[str, float]:
    cleaned = simple_clean(text)
    tokens = cleaned.split()
    if not tokens:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    neu = max(0, len(tokens) - pos - neg)
    total = pos + neg + neu
    pos_prop = pos / total if total else 0.0
    neg_prop = neg / total if total else 0.0
    neu_prop = neu / total if total else 1.0
    compound = pos_prop - neg_prop
    return {"neg": neg_prop, "neu": neu_prop, "pos": pos_prop, "compound": compound}


def compound_to_label(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


# -------------------------
# Analyzer detection (lazy imports)
# -------------------------
def detect_sentiment_analyzer() -> Dict[str, object]:
    """
    Try NLTK VADER, then vaderSentiment, otherwise fallback.
    Returns a dict with 'score_fn' and 'name'.
    """
    # Try NLTK VADER
    try:
        import nltk  # type: ignore
        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError:
            try:
                nltk.download("vader_lexicon", quiet=True)
                nltk.data.find("sentiment/vader_lexicon")
            except Exception:
                raise ImportError("vader_lexicon not available")
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        analyzer = SentimentIntensityAnalyzer()
        return {"score_fn": lambda t: analyzer.polarity_scores(t), "name": "NLTK VADER (nltk + vader_lexicon)"}
    except Exception:
        # Try vaderSentiment
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            analyzer = SentimentIntensityAnalyzer()
            return {"score_fn": lambda t: analyzer.polarity_scores(t), "name": "vaderSentiment package"}
        except Exception:
            return {"score_fn": fallback_sentiment, "name": "Fallback keyword-based analyzer (approx.)"}


# -------------------------
# App UI & flow
# -------------------------
with st.sidebar:
    st.header("Options & Inputs")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], help="CSV with a text column, e.g. review_content")
    use_sample = st.button("Use sample data")
    text_col_input = st.text_input("Text column name (auto-detect if left default)", value="review_content")
    sample_n = st.number_input("Sample size (0 = all)", min_value=0, value=0, step=1)
    want_wordcloud = st.checkbox("Enable WordClouds (if environment has `wordcloud`)", value=True)
    remove_numbers = st.checkbox("Remove numbers during preprocessing", value=True)
    lower_case = st.checkbox("Lowercase text during preprocessing", value=True)
    run = st.button("Analyze")

SAMPLE_CSV = """review_content
"Good product, arrived on time, works as described."
"Terrible packaging. Cable stopped working after two days."
"Average experience — nothing special, not worth extra cost."
"Excellent! Very fast delivery and works perfectly."
"Not as described. Missing parts and poor quality."
"Love it! Exactly as advertised."
"Stopped working after a week, disappointed."
"""

def load_sample_df() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(SAMPLE_CSV))


def read_uploaded_csv(file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None


if uploaded is None and not use_sample:
    st.info("Upload a CSV or press 'Use sample data' to start.")
    st.stop()

df = load_sample_df() if use_sample else read_uploaded_csv(uploaded)
if df is None:
    st.stop()

available_cols = list(df.columns)
if text_col_input and text_col_input in available_cols:
    text_col = text_col_input
else:
    candidates = ["review_content", "review", "text", "comment", "content"]
    text_col = next((c for c in candidates if c in available_cols), available_cols[0] if available_cols else None)

if text_col is None:
    st.error("No text column found in uploaded CSV.")
    st.stop()

st.sidebar.success(f"Using text column: {text_col}")
st.subheader("Data preview")
st.dataframe(df.head(8))

if not run:
    st.info("Adjust options and click Analyze to run.")
    st.stop()

# Detect analyzer
det = detect_sentiment_analyzer()
score_fn: Callable[[str], Dict[str, float]] = det["score_fn"]  # type: ignore
analyzer_name = det["name"]
st.success(f"Analyzer selected: {analyzer_name}")
if analyzer_name != "NLTK VADER (nltk + vader_lexicon)":
    st.info(
        "Tip: Add `nltk` and ensure the `vader_lexicon` resource (or install `vaderSentiment`) in your requirements "
        "to get higher-quality VADER scoring. The app will automatically use it when available."
    )

# WordCloud gating
if want_wordcloud and not _HAS_WORDCLOUD:
    st.warning(
        "WordCloud requested but `wordcloud` package is not installed in this environment. "
        "WordClouds will be disabled. To enable, add `wordcloud` and `pillow` to requirements and redeploy."
    )
    want_wordcloud = False

# Optional sampling
if sample_n and 0 < sample_n < len(df):
    df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

# Process rows
n = len(df)
progress = st.progress(0)
status = st.empty()
clean_texts = []
scores_list = []
for i, raw in enumerate(df[text_col].astype(str)):
    status.text(f"Processing {i+1}/{n} ...")
    cleaned = simple_clean(raw, remove_numbers=remove_numbers, lower=lower_case)
    clean_texts.append(cleaned)
    try:
        sc = score_fn(cleaned)
        # normalize to expected keys
        if not all(k in sc for k in ("neg", "neu", "pos", "compound")):
            sc = {
                "neg": float(sc.get("neg", 0.0)),
                "neu": float(sc.get("neu", 0.0)),
                "pos": float(sc.get("pos", 0.0)),
                "compound": float(sc.get("compound", sc.get("compound_score", 0.0)))
            }
    except Exception:
        sc = fallback_sentiment(cleaned)
    scores_list.append(sc)
    progress.progress((i + 1) / n)
progress.empty()
status.empty()

scores_df = pd.DataFrame(scores_list)
results = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
results["_clean_text"] = clean_texts
results["_label"] = results["compound"].apply(compound_to_label)

# Overview: distribution
st.subheader("Overview")
counts = results["_label"].value_counts().reset_index()
counts.columns = ["label", "count"]
counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(1)

col1, col2 = st.columns([2, 1])
with col1:
    if counts.empty:
        st.warning("No sentiment labels to display.")
    else:
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("label:N", title="Sentiment"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("label:N", legend=None)
        ).properties(title="Sentiment distribution")
        st.altair_chart(chart, width="stretch")
with col2:
    st.metric("Total reviews", len(results))
    for _, row in counts.iterrows():
        st.write(f"{row['label'].capitalize()}: {int(row['count'])} ({row['percent']}%)")

st.markdown("---")

# Helper to compute top words
def top_words_from_texts(texts, top_n: int = 20, stopwords=None) -> Tuple[Counter, list]:
    stop = set(_BASIC_STOPWORDS) if stopwords is None else set(stopwords)
    tokens = []
    for t in texts:
        for tok in t.split():
            tok = tok.strip().lower()
            if len(tok) <= 1 or tok in stop:
                continue
            tokens.append(tok)
    ctr = Counter(tokens)
    return ctr, ctr.most_common(top_n)


# Tabs for each sentiment with details
tab_overview, tab_pos, tab_neu, tab_neg = st.tabs(["Overview (again)", "Positive", "Neutral", "Negative"])

# Reuse counts and results in tabs
with tab_overview:
    st.write("Quick overview and ability to download full results.")
    st.dataframe(results[[text_col, "_clean_text", "_label", "compound"]].head(10))
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download full results CSV", data=csv_bytes, file_name="sentiment_results_full.csv", mime="text/csv")

def render_sentiment_section(label: str, df_results: pd.DataFrame):
    sub = df_results[df_results["_label"] == label]
    count = len(sub)
    pct = (count / len(df_results) * 100) if len(df_results) else 0.0
    st.header(f"{label.capitalize()} — {count} reviews ({pct:.1f}%)")
    if count == 0:
        st.warning(f"No {label} reviews found.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Sample reviews")
        st.dataframe(sub[[text_col, "_clean_text", "compound"]].head(8))
        st.download_button(
            f"Download {label} reviews CSV",
            data=sub.to_csv(index=False).encode("utf-8"),
            file_name=f"{label}_reviews.csv",
            mime="text/csv",
        )
    with c2:
        st.subheader("Top words")
        ctr, most = top_words_from_texts(sub["_clean_text"].astype(str), top_n=15)
        if most:
            top_df = pd.DataFrame(most, columns=["word", "count"])
            bar = alt.Chart(top_df).mark_bar().encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("word:N", sort="-x", title="Word"),
                tooltip=["word", "count"]
            ).properties(height=300)
            st.altair_chart(bar, width="stretch")
        else:
            st.write("No significant words found.")

    # WordCloud (if available)
    if want_wordcloud:
        if not _HAS_WORDCLOUD:
            st.info("WordCloud not available in this environment.")
        else:
            st.subheader("Word Cloud")
            combined = " ".join(sub["_clean_text"].astype(str))
            if not combined.strip():
                st.warning("No text available for word cloud.")
            else:
                wc = WordCloud(width=1200, height=600, background_color="white", colormap="Blues")
                wc.generate(combined)
                img = wc.to_image()
                st.image(img, caption=f"WordCloud for {label} reviews", use_column_width=True)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                st.download_button(f"Download {label} WordCloud (PNG)", data=buf, file_name=f"{label}_wordcloud.png", mime="image/png")

with tab_pos:
    render_sentiment_section("positive", results)

with tab_neu:
    render_sentiment_section("neutral", results)

with tab_neg:
    render_sentiment_section("negative", results)

st.info(
    "Notes:\n"
    "- WordCloud requires the `wordcloud` package to be installed on the host. If WordClouds do not appear, "
    "add `wordcloud` and `pillow` to requirements.txt and redeploy (may require compilation on some platforms).\n"
    "- For higher-quality sentiment scoring, install `nltk` and ensure the `vader_lexicon` resource (or install `vaderSentiment`)."
)