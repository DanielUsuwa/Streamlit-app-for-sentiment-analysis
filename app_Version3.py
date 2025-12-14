#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Sentiment Analyzer — lightweight, robust, no external NLP deps, no wordcloud.

This version:
- DOES NOT import nltk, wordcloud, or any optional packages that may not be available on the host.
- Uses a simple keyword-based sentiment analyzer so the app always runs reliably.
- Uses Altair for charts and Streamlit APIs compatible with newer versions (width='stretch').
- Is UX-friendly with clear messages, progress indicators, CSV download, and column auto-detection.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import io
import re
from typing import Dict, Optional, Callable

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Sentiment Analyzer (lightweight)", layout="wide")

# -------------------------
# Lightweight preprocessing + analyzer
# -------------------------
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_HTML = re.compile(r'<.*?>')
RE_NON_ALPHANUM = re.compile(r'[^A-Za-z0-9\s]')
RE_MULTI_WS = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\d+')

_POS_WORDS = {
    "good", "great", "excellent", "love", "nice", "fast", "happy", "recommend",
    "perfect", "best", "works", "working", "satisfied", "okay", "awesome", "amazing"
}
_NEG_WORDS = {
    "bad", "poor", "terrible", "hate", "slow", "worse", "worst", "broken",
    "disappoint", "missing", "problem", "return", "refund", "awful"
}


def simple_clean(text: str, remove_numbers: bool = True, lower: bool = True) -> str:
    """Minimal deterministic cleaning safe for all environments."""
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
    """
    Return scores compatible with VADER-style output:
    {'neg':..., 'neu':..., 'pos':..., 'compound': ...}
    This is a simple keyword-based heuristic to ensure the app always runs.
    """
    cleaned = simple_clean(text)
    tokens = cleaned.split()
    if not tokens:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    neu = max(0, len(tokens) - pos - neg)

    # normalized proportions
    total = pos + neg + neu
    if total == 0:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    pos_prop = pos / total
    neg_prop = neg / total
    neu_prop = neu / total

    # compound heuristic: difference between pos and neg proportion (in [-1, 1])
    compound = pos_prop - neg_prop

    return {"neg": neg_prop, "neu": neu_prop, "pos": pos_prop, "compound": compound}


def compound_to_label(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


# -------------------------
# Streamlit UI
# -------------------------
st.title("Sentiment Analyzer (lightweight, always-on)")
st.markdown(
    "This app uses a small, deterministic keyword-based analyzer so it runs reliably on any host. "
    "It doesn't require external NLP packages. Results are quick but approximate — suitable for exploratory use."
)

with st.sidebar:
    st.header("Options")
    uploaded = st.file_uploader("Upload CSV file (must contain a text column)", type=["csv"])
    use_sample = st.button("Use sample data")
    text_col_input = st.text_input("Text column name (leave default to auto-detect)", value="review_content")
    sample_n = st.number_input("Sample size (0 = all)", min_value=0, value=0, step=1)
    remove_numbers = st.checkbox("Remove numbers during preprocessing", value=True)
    lower_case = st.checkbox("Lowercase text during preprocessing", value=True)
    run = st.button("Analyze")

SAMPLE_CSV = """review_content
"Good product, arrived on time, works as described."
"Terrible packaging. Cable stopped working after two days."
"Average experience — nothing special, not worth extra cost."
"Excellent! Very fast delivery and works perfectly."
"Not as described. Missing parts and poor quality."
"""

def load_sample_df() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(SAMPLE_CSV))


def read_uploaded_csv(up) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(up)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None


# Wait for user to provide data
if uploaded is None and not use_sample:
    st.info("Upload a CSV or press 'Use sample data' in the sidebar to begin.")
    st.stop()

# Load data
df = load_sample_df() if use_sample else read_uploaded_csv(uploaded)
if df is None:
    st.stop()

# Choose text column
available_cols = list(df.columns)
if text_col_input and text_col_input in available_cols:
    text_col = text_col_input
else:
    candidates = ["review_content", "review", "text", "content", "comment"]
    text_col = next((c for c in candidates if c in available_cols), available_cols[0] if available_cols else None)

if text_col is None:
    st.error("No columns found in the uploaded CSV.")
    st.stop()

st.sidebar.success(f"Using text column: {text_col}")

st.subheader("Data preview")
st.dataframe(df.head(8))

if not run:
    st.info("Click 'Analyze' to run the lightweight sentiment analysis.")
    st.stop()

# Optionally sample
if sample_n and 0 < sample_n < len(df):
    df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

# Processing loop with progress
n = len(df)
progress = st.progress(0)
status = st.empty()

clean_texts = []
scores = []
for i, t in enumerate(df[text_col].astype(str)):
    status.text(f"Processing {i+1}/{n} ...")
    cleaned = simple_clean(t, remove_numbers=remove_numbers, lower=lower_case)
    clean_texts.append(cleaned)
    sc = fallback_sentiment(cleaned)
    scores.append(sc)
    progress.progress((i + 1) / n)

progress.empty()
status.empty()

# Build results
scores_df = pd.DataFrame(scores)
results = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
results["_clean_text"] = clean_texts
results["_label"] = results["compound"].apply(compound_to_label)

st.success("Analysis complete")
st.subheader("Sample results")
st.dataframe(results[[text_col, "_clean_text", "_label", "compound"]].head(10))

# Sentiment distribution chart (Altair)
counts = results["_label"].value_counts().reset_index()
counts.columns = ["label", "count"]
if counts.empty:
    st.warning("No sentiment labels to display.")
else:
    bar = alt.Chart(counts).mark_bar().encode(
        x=alt.X("label:N", title="Sentiment"),
        y=alt.Y("count:Q", title="Count"),
        color=alt.Color("label:N", legend=None)
    ).properties(title="Sentiment distribution")
    st.altair_chart(bar, width="stretch")

# Download results
csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

st.info("Note: this app uses a lightweight keyword-based analyzer for reliability and immediate availability.")