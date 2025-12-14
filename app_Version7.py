#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Sentiment Analyzer — improved desktop layout, compact mobile, safe WordCloud fallback.

Changes in this version:
- Removed the "Quick, mobile-friendly..." heading sentence.
- Set Streamlit layout to wide for better desktop use while remaining mobile-friendly.
- Increased chart sizes and used a wider main column to improve desktop UX.
- Fixed Altair color scheme error (replaced invalid scheme with 'blues').
- Wrapped fallback word-cloud rendering in a safe try/except to avoid Altair SchemaValidationError.
- Kept WordCloud image generation when the package is available; otherwise a compact notice + optional fallback visual cloud.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from typing import Callable, Dict, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Guarded WordCloud import
try:
    from wordcloud import WordCloud  # type: ignore
    _HAS_WORDCLOUD = True
except Exception:
    WordCloud = None  # type: ignore
    _HAS_WORDCLOUD = False

# Minimal stopwords for frequency calculations
_BASIC_STOPWORDS = {
    "the", "and", "is", "in", "it", "of", "to", "a", "this", "that", "for", "on", "with",
    "was", "are", "as", "but", "they", "i", "we", "you", "have", "has", "been", "not",
    "be", "at", "or", "an", "so", "if", "its", "by", "from"
}

# Page config: wide layout for better desktop experience
st.set_page_config(page_title="Sentiment Analyzer", layout="wide", initial_sidebar_state="collapsed")
st.title("Sentiment Analyzer")
# Note: removed extra introductory sentence to keep the UI concise.

# -------------------------
# Helpers: cleaning, fallback sentiment, analyzer detection
# -------------------------
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_HTML = re.compile(r'<.*?>')
RE_NON_ALPHANUM = re.compile(r'[^A-Za-z0-9\s]')
RE_MULTI_WS = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\d+')

_POS_WORDS = {
    "good", "great", "excellent", "love", "nice", "fast", "happy", "recommend", "perfect", "best",
    "works", "working", "satisfied", "okay", "awesome", "amazing", "delivered", "quick"
}
_NEG_WORDS = {
    "bad", "poor", "terrible", "hate", "slow", "worse", "worst", "broken", "disappoint",
    "missing", "problem", "return", "refund", "awful", "damage", "delay", "not"
}


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


def fallback_sentiment(text: str) -> Dict[str, float]:
    cleaned = simple_clean(text)
    tokens = cleaned.split()
    if not tokens:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    neu = max(0, len(tokens) - pos - neg)
    total = pos + neg + neu or 1
    pos_prop = pos / total
    neg_prop = neg / total
    neu_prop = neu / total
    compound = pos_prop - neg_prop
    return {"neg": neg_prop, "neu": neu_prop, "pos": pos_prop, "compound": compound}


def compound_to_label(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


def detect_sentiment_analyzer() -> Dict[str, object]:
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
        return {"score_fn": lambda t: analyzer.polarity_scores(t), "name": "NLTK VADER"}
    except Exception:
        # Try vaderSentiment
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            analyzer = SentimentIntensityAnalyzer()
            return {"score_fn": lambda t: analyzer.polarity_scores(t), "name": "vaderSentiment"}
        except Exception:
            return {"score_fn": fallback_sentiment, "name": "Fallback (keyword)"}


# -------------------------
# Controls (top row) - kept compact for mobile, horizontally arranged and suitable on desktop
# -------------------------
controls = st.container()
with controls:
    cols = st.columns([1.2, 1, 1, 1])
    with cols[0]:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="CSV with a text column (e.g. review_content)")
    with cols[1]:
        use_sample = st.button("Use sample")
    with cols[2]:
        analyze_btn = st.button("Analyze", type="primary")
    with cols[3]:
        # compact text column input
        text_col_input = st.text_input("Text column", value="review_content")

    cols2 = st.columns([1, 1, 1, 1])
    with cols2[0]:
        sample_n = st.number_input("Sample N (0=all)", min_value=0, value=0, step=1)
    with cols2[1]:
        wc_toggle = st.checkbox("WordCloud (image)", value=True)
    with cols2[2]:
        remove_numbers = st.checkbox("Remove numbers", value=True)
    with cols2[3]:
        lower_case = st.checkbox("Lowercase", value=True)

    # Advanced options tucked away
    with st.expander("Advanced (hidden)", expanded=False):
        show_fallback_cloud = st.checkbox("Show fallback cloud if WordCloud unavailable", value=True)
        # Keep help minimal and hidden by default
        st.markdown("Compact help is available in the 'Help & notes' expander at the bottom of the page.")


# Sample CSV (small)
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


def read_uploaded_csv(f) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(f)
    except Exception:
        return None


# Load data
if uploaded is None and not use_sample:
    st.info("Upload CSV or tap 'Use sample' then tap Analyze.")
    st.stop()

df = load_sample_df() if use_sample else read_uploaded_csv(uploaded)
if df is None:
    st.error("Failed to read CSV. Ensure it is valid and has a header row.")
    st.stop()

# Choose text column
cols = list(df.columns)
if text_col_input and text_col_input in cols:
    text_col = text_col_input
else:
    candidates = ["review_content", "review", "text", "comment", "content"]
    text_col = next((c for c in candidates if c in cols), cols[0] if cols else None)

if text_col is None:
    st.error("No columns found in CSV.")
    st.stop()

# Don't run until Analyze pressed
if not analyze_btn:
    st.info("Ready. Tap Analyze to run.")
    st.stop()

# Prepare analyzer
det = detect_sentiment_analyzer()
score_fn: Callable[[str], Dict[str, float]] = det["score_fn"]  # type: ignore
analyzer_name = det["name"]

# Compact analyzer notice (single-line)
st.caption(f"Analyzer: {analyzer_name}")

# Optionally apply sampling
if sample_n and 0 < sample_n < len(df):
    df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

# Process with progress
n = len(df)
progress = st.progress(0)
status = st.empty()
clean_texts = []
scores = []
for i, raw in enumerate(df[text_col].astype(str)):
    status.text(f"{i+1}/{n}")
    cleaned = simple_clean(raw, remove_numbers=remove_numbers, lower=lower_case)
    clean_texts.append(cleaned)
    try:
        sc = score_fn(cleaned)
        if not all(k in sc for k in ("neg", "neu", "pos", "compound")):
            sc = {
                "neg": float(sc.get("neg", 0.0)),
                "neu": float(sc.get("neu", 0.0)),
                "pos": float(sc.get("pos", 0.0)),
                "compound": float(sc.get("compound", sc.get("compound_score", 0.0)))
            }
    except Exception:
        sc = fallback_sentiment(cleaned)
    scores.append(sc)
    progress.progress((i + 1) / n)
progress.empty()
status.empty()

scores_df = pd.DataFrame(scores)
results = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
results["_clean_text"] = clean_texts
results["_label"] = results["compound"].apply(compound_to_label)

# Layout: wider main area on desktop using columns
main_col, side_col = st.columns([3, 1])

# Overview and distribution in main column
with main_col:
    st.subheader("Distribution")
    counts = results["_label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    if counts.empty:
        st.warning("No sentiment labels to display.")
    else:
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("label:N", title="Sentiment"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("label:N", legend=None, scale=alt.Scale(scheme="blues"))
        ).properties(height=300, title="Sentiment distribution")
        st.altair_chart(chart, width="stretch")

    # Tabs for per-sentiment details
    tab_overview, tab_pos, tab_neu, tab_neg = st.tabs(["Overview", "Positive", "Neutral", "Negative"])

# Side column: quick metrics
with side_col:
    st.metric("Total reviews", len(results))
    for _, r in counts.iterrows():
        st.write(f"{r['label'].capitalize()}: {int(r['count'])} ({r['percent']}%)")
    st.download_button("Download results CSV", data=results.to_csv(index=False).encode("utf-8"),
                       file_name="sentiment_results.csv")

# Helper for top words
def top_words(texts, top_n=15, stopwords=None):
    stop = set(_BASIC_STOPWORDS) if stopwords is None else set(stopwords)
    ctr = Counter()
    for t in texts:
        for tok in t.split():
            tok = tok.lower().strip()
            if len(tok) <= 1 or tok in stop:
                continue
            ctr[tok] += 1
    return ctr.most_common(top_n)

# Fallback visual wordcloud renderer (safe)
def render_fallback_wordcloud(texts, title="Word Cloud (fallback)", max_words=60):
    try:
        freq = top_words(texts, top_n=max_words)
        if not freq:
            st.write("No words available.")
            return
        df_wc = pd.DataFrame(freq, columns=["word", "count"])
        max_c = df_wc["count"].max()
        min_c = df_wc["count"].min()
        def scale(c, lo=12, hi=48):
            if max_c == min_c:
                return (lo+hi)/2
            return lo + (hi-lo) * ((c - min_c) / (max_c - min_c))
        df_wc["size"] = df_wc["count"].apply(scale)
        rng = np.random.default_rng(42)
        df_wc["x"] = rng.random(len(df_wc))
        df_wc["y"] = rng.random(len(df_wc))
        chart = (
            alt.Chart(df_wc)
            .mark_text()
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                text=alt.Text("word:N"),
                size=alt.Size("size:Q", legend=None),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["word", "count"]
            )
            .properties(height=420, title=title)
        )
        st.altair_chart(chart, width="stretch")
    except Exception:
        # If something goes wrong with Altair, show a bar chart fallback
        st.warning("Couldn't render visual cloud; showing top words instead.")
        freq = top_words(texts, top_n=20)
        if freq:
            df_top = pd.DataFrame(freq, columns=["word", "count"])
            bar = alt.Chart(df_top).mark_bar().encode(
                x=alt.X("count:Q"),
                y=alt.Y("word:N", sort="-x"),
                tooltip=["word", "count"]
            ).properties(height=300)
            st.altair_chart(bar, width="stretch")
        else:
            st.write("No words to display.")

# Per-sentiment section rendering
def sentiment_tab(label: str, tab_area):
    with tab_area:
        sub = results[results["_label"] == label]
        st.header(f"{label.capitalize()} ({len(sub)})")
        if sub.empty:
            st.write("No reviews in this category.")
            return
        st.subheader("Sample")
        st.dataframe(sub[[text_col, "_clean_text", "compound"]].head(8))
        st.subheader("Top words")
        most = top_words(sub["_clean_text"].astype(str), top_n=12)
        if most:
            df_top = pd.DataFrame(most, columns=["word", "count"])
            bar = alt.Chart(df_top).mark_bar().encode(
                x=alt.X("count:Q"),
                y=alt.Y("word:N", sort="-x"),
                tooltip=["word", "count"]
            ).properties(height=260)
            st.altair_chart(bar, width="stretch")
        else:
            st.write("No top words found.")
        # WordCloud area
        if wc_toggle:
            if _HAS_WORDCLOUD:
                combined = " ".join(sub["_clean_text"].astype(str).tolist())
                if combined.strip():
                    wc = WordCloud(width=1200, height=600, background_color="white", colormap="Blues")
                    wc.generate(combined)
                    img = wc.to_image()
                    st.image(img, use_column_width=True)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(f"Download {label} wordcloud", data=buf, file_name=f"{label}_wordcloud.png")
                else:
                    st.write("No text to build word cloud.")
            else:
                if show_fallback_cloud:
                    texts = sub["_clean_text"].astype(str).tolist()
                    render_fallback_wordcloud(texts, title=f"{label.capitalize()} (fallback cloud)")
                else:
                    st.caption("WordCloud image unavailable. Enable fallback cloud in Advanced options.")

# Attach tabs
sentiment_tab("positive", tab_pos)
sentiment_tab("neutral", tab_neu)
sentiment_tab("negative", tab_neg)

# Overview tab
with tab_overview:
    st.subheader("Overview")
    st.dataframe(results[[text_col, "_clean_text", "_label", "compound"]].head(10))
    st.download_button("Download full results CSV", data=results.to_csv(index=False).encode("utf-8"),
                       file_name="sentiment_results_full.csv")

# Minimal Help (collapsed)
with st.expander("Help & notes", expanded=False):
    st.write("- Use the top controls to upload data and run analysis.")
    st.write("- WordCloud images require the `wordcloud` package. If images don't show, a compact notice is displayed.")
    st.write("- For higher-quality sentiment scoring, deploy with `nltk` (and ensure `vader_lexicon`) or add `vaderSentiment` to requirements.")
