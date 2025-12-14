#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit app (version safe) for sentiment analysis (VADER) with preprocessing and optional word cloud.
This version guards optional imports (wordcloud) so the app won't crash when the package is unavailable.
"""
from __future__ import annotations

import io
import os
import re
from typing import Optional

import pandas as pd
import streamlit as st

# Visualization (use Altair for charts)
import altair as alt
from PIL import Image  # pillow is commonly available

# Optional WordCloud import - guarded so missing package doesn't crash the app
try:
    from wordcloud import WordCloud  # type: ignore
    _HAS_WORDCLOUD = True
except Exception:
    WordCloud = None  # type: ignore
    _HAS_WORDCLOUD = False

# NLTK and VADER
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -----------------------------
# Utilities (preprocessing + NLTK)
# -----------------------------

RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_HTML = re.compile(r'<.*?>')
RE_NON_ALPHANUM = re.compile(r'[^A-Za-z0-9\s]')
RE_MULTI_WS = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\d+')

def ensure_nltk_resources():
    """
    Ensure necessary NLTK resources are available. Downloads are idempotent.
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "vader_lexicon": "sentiment/vader_lexicon",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            with st.spinner(f"Downloading NLTK resource: {name} ..."):
                nltk.download(name, quiet=True)

def _nltk_pos_to_wordnet_pos(nltk_pos_tag: str):
    if nltk_pos_tag.startswith('J'):
        return wordnet.ADJ
    if nltk_pos_tag.startswith('V'):
        return wordnet.VERB
    if nltk_pos_tag.startswith('N'):
        return wordnet.NOUN
    if nltk_pos_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def preprocess_text(text: str,
                    stop_words_set: set,
                    lemmatizer: WordNetLemmatizer,
                    remove_numbers: bool = True,
                    keep_case: bool = False) -> str:
    """Clean, tokenize, remove stop words and lemmatize."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = RE_HTML.sub(" ", text)
    text = RE_URL.sub(" ", text)
    if not keep_case:
        text = text.lower()
    if remove_numbers:
        text = RE_NUMBERS.sub(" ", text)
    text = RE_NON_ALPHANUM.sub(" ", text)
    text = RE_MULTI_WS.sub(" ", text)
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    cleaned = []
    for token, pos in pos_tags:
        tok = token.lower()
        if tok in stop_words_set or len(tok) <= 1:
            continue
        wn_pos = _nltk_pos_to_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(tok, pos=wn_pos)
        cleaned.append(lemma)
    return " ".join(cleaned)

def label_from_compound(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Simple Sentiment Analyzer", layout="wide", initial_sidebar_state="expanded")

st.title("Simple Sentiment Analyzer")
st.markdown(
    "Upload a CSV of text reviews, select the text column, and get sentiment labels (VADER), "
    "cleaned text, a sentiment distribution chart, and an optional word cloud for positive reviews."
)

with st.sidebar:
    st.header("Inputs & Options")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV with a text column (e.g. review_content)")
    use_sample = st.button("Use sample data")
    text_col_input = st.text_input("Text column name (will auto-detect common names)", value="review_content")
    sample_size = st.slider("Sample (0 = use all)", min_value=0, max_value=5000, value=0, step=10)
    # Word cloud checkbox will be disabled if wordcloud package is missing
    if _HAS_WORDCLOUD:
        generate_wordcloud = st.checkbox("Generate word cloud for positive reviews", value=True)
    else:
        generate_wordcloud = False
        st.checkbox("Generate word cloud for positive reviews (unavailable)", value=False, disabled=True)
        st.markdown(
            "⚠️ Word cloud feature is disabled because the `wordcloud` package is not installed on this environment. "
            "To enable it, add `wordcloud` and `pillow` to your `requirements.txt` and redeploy, if supported."
        )
    remove_numbers = st.checkbox("Remove numbers during preprocessing", value=True)
    keep_case = st.checkbox("Keep original casing (disable lowercasing)", value=False)
    run_button = st.button("Analyze")
    st.markdown("---")
    st.markdown("Help / Troubleshooting:")
    st.info(
        "If the app stalls during first run, it's likely downloading NLTK resources. The app shows a spinner while doing so."
    )

# Load sample data baked into repo (small)
SAMPLE_CSV = """review_content
"Good product, arrived on time, works as described."
"Terrible packaging. Cable stopped working after two days."
"Average experience — nothing special, not worth extra cost."
"Excellent! Very fast delivery and works perfectly."
"Not as described. Missing parts and poor quality."
"""

def load_sample_df() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(SAMPLE_CSV))

def read_uploaded_csv(uploaded) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

# Main interactive flow
if uploaded_file is None and not use_sample:
    st.info("Upload a CSV or click 'Use sample data' to try the app.")
    st.stop()

# Prepare data
if use_sample:
    df = load_sample_df()
else:
    df = read_uploaded_csv(uploaded_file)
    if df is None:
        st.stop()

# Auto-detect text column
common_names = ["review_content", "review", "text", "content", "comment"]
available = list(df.columns)
if text_col_input and text_col_input in available:
    text_col = text_col_input
else:
    text_col = next((c for c in common_names if c in available), available[0] if available else None)

if text_col is None:
    st.error("Could not find any columns in the uploaded CSV.")
    st.stop()

st.sidebar.success(f"Using text column: {text_col}")

# Show a preview
st.subheader("Data preview")
st.dataframe(df.head(10))

# Run analysis when user presses Run
if not run_button:
    st.info("Adjust options and click 'Analyze' to run sentiment analysis.")
    st.stop()

# Analysis starts here
try:
    ensure_nltk_resources()
except Exception as e:
    st.error(f"Failed to ensure NLTK resources: {e}")
    st.stop()

# Prepare NLTK helpers
stop_words_set = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Optionally sample
if sample_size and sample_size > 0 and sample_size < len(df):
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

# Progress UI
progress_bar = st.progress(0)
status_text = st.empty()

clean_texts = []
sent_scores = []
n = len(df)
for i, text in enumerate(df[text_col].astype(str)):
    status_text.text(f"Preprocessing and scoring row {i+1}/{n} ...")
    try:
        clean = preprocess_text(text, stop_words_set=stop_words_set, lemmatizer=lemmatizer,
                                remove_numbers=remove_numbers, keep_case=keep_case)
    except Exception:
        # Fallback: minimal cleaning
        clean = re.sub(r'\s+', ' ', str(text)).strip()
    clean_texts.append(clean)
    scores = analyzer.polarity_scores(clean)
    scores["label"] = label_from_compound(scores["compound"])
    sent_scores.append(scores)
    progress_bar.progress((i + 1) / n)

progress_bar.empty()
status_text.empty()

# Attach results
scores_df = pd.DataFrame(sent_scores)
results = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
results["_clean_text"] = clean_texts

st.success("Analysis complete!")
st.subheader("Sentiment summary")

# Distribution chart using Altair (no plotly required)
counts = results["label"].value_counts().reset_index()
counts.columns = ["label", "count"]
if len(counts) == 0:
    st.warning("No sentiment labels to display.")
else:
    chart = alt.Chart(counts).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color("label:N", legend=alt.Legend(title="Sentiment")),
        tooltip=["label", "count"]
    ).properties(title="Sentiment distribution")
    st.altair_chart(chart, use_container_width=True)

# Show sample results
st.subheader("Sample results")
st.dataframe(results[[text_col, "_clean_text", "label", "compound"]].head(10))

# Download results CSV
def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

csv_bytes = convert_df_to_csv_bytes(results)
st.download_button("Download results CSV", data=csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

# Word cloud (only if WordCloud is available)
if generate_wordcloud:
    if not _HAS_WORDCLOUD:
        st.warning("Word cloud generation is not available in this environment (wordcloud package missing).")
    else:
        pos_text = " ".join(results.loc[results["label"] == "positive", "_clean_text"].astype(str))
        if pos_text.strip() == "":
            st.warning("No positive reviews to build a word cloud.")
        else:
            wc = WordCloud(width=1200, height=600, background_color="white", colormap="Blues")
            wc.generate(pos_text)
            img = wc.to_image()
            st.subheader("Word cloud (positive reviews)")
            st.image(img, use_column_width=True)
            # Provide download of image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download word cloud (PNG)", data=buf, file_name="positive_wordcloud.png", mime="image/png")

st.info("If something goes wrong, check the README for troubleshooting tips.")