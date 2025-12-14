#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Sentiment Analyzer — dataset preview + theme toggle (light/dark)

Changes:
- Adds a 'Dataset preview' expander with adjustable number of rows and an option to show the full dataset.
- Adds a Theme selector (Auto / Light / Dark) with CSS injection to switch background/text colors.
- Keeps previous features: up to 3 text columns, WordCloud (if installed) with fallback visual cloud.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from typing import Callable, Dict, Optional

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
# Theme helper (Auto / Light / Dark)
# -------------------------
def apply_theme_css(theme_choice: str) -> None:
    """
    Inject CSS to set a coherent background and text color for Light or Dark theme.
    'Auto' does nothing and leaves Streamlit default.
    """
    if theme_choice == "Auto":
        return
    if theme_choice == "Light":
        bg = "#FFFFFF"
        text = "#0f1724"
        panel = "#F7FAFF"
    else:  # Dark
        bg = "#0b1220"
        text = "#E6EEF8"
        panel = "#0f1724"

    css = f"""
    <style>
    /* Main background */
    .stApp > .main {{
        background-color: {bg};
        color: {text};
    }}
    /* Block containers */
    .block-container {{
        background-color: {bg};
        color: {text};
    }}
    /* DataFrame cells */
    .stDataFrame td, .stDataFrame th {{
        background-color: {panel} !important;
        color: {text} !important;
    }}
    /* Metric and other widgets */
    .stMetric .main .value {{
        color: {text} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -------------------------
# Controls (top rows)
# -------------------------
controls = st.container()
with controls:
    # First row: upload, sample, analyze, theme select
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    with c1:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="CSV with text columns such as review_content")
    with c2:
        use_sample = st.button("Use sample")
    with c3:
        analyze_btn = st.button("Analyze", type="primary")
    with c4:
        theme_choice = st.selectbox("Theme", ["Auto", "Light", "Dark"], index=0)

    # Second row: three text column inputs + sample size
    r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
    with r1:
        text_col1 = st.text_input("Text column 1", value="review_content")
    with r2:
        text_col2 = st.text_input("Text column 2 (optional)", value="")
    with r3:
        text_col3 = st.text_input("Text column 3 (optional)", value="")
    with r4:
        sample_n = st.number_input("Sample N (0=all)", min_value=0, value=0, step=1)

    # Third row: toggles
    t1, t2, t3, t4 = st.columns([1, 1, 1, 1])
    with t1:
        wc_toggle = st.checkbox("WordCloud (image)", value=True)
    with t2:
        remove_numbers = st.checkbox("Remove numbers", value=True)
    with t3:
        lower_case = st.checkbox("Lowercase", value=True)
    with t4:
        show_preview_all = st.checkbox("Show full dataset preview on complete", value=False)

    # Advanced tucked away but minimal
    with st.expander("Advanced (hidden)", expanded=False):
        show_fallback_cloud = st.checkbox("Show fallback cloud if WordCloud unavailable", value=True)
        st.markdown("Compact help available in the 'Help & notes' expander at the bottom.")


# Apply theme CSS (if Light or Dark)
apply_theme_css(theme_choice)

# Sample CSV (small)
SAMPLE_CSV = """review_content,other_comment,followup
"Good product, arrived on time, works as described.","Fast delivery","Will buy again"
"Terrible packaging. Cable stopped working after two days.","Packaging bad","Returned"
"Average experience — nothing special, not worth extra cost.","No followup",""
"Excellent! Very fast delivery and works perfectly.","Great service","Recommended"
"Not as described. Missing parts and poor quality.","Missing parts","Refund requested"
"Love it! Exactly as advertised.","Happy",""
"Stopped working after a week, disappointed.","Broke quickly","Contacted support"
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
    st.error("Failed to read CSV. Ensure it is a valid CSV with headers.")
    st.stop()

# Resolve selected columns: ensure they exist in the dataframe
available_cols = list(df.columns)
selected_cols = []
# Text column 1 required (try auto-detect if missing)
if text_col1 and text_col1 in available_cols:
    selected_cols.append(text_col1)
else:
    candidates = ["review_content", "review", "text", "comment", "content"]
    detected = next((c for c in candidates if c in available_cols), None)
    if detected:
        selected_cols.append(detected)
    else:
        st.error("Could not find a suitable text column. Please specify Text column 1.")
        st.stop()

# Optional columns 2 & 3
for tc in (text_col2, text_col3):
    if tc and tc.strip() and tc in available_cols:
        selected_cols.append(tc)

# Show which columns are used (compact)
st.caption(f"Using text columns: {', '.join(selected_cols)}")

# Build combined text series
combined_col = "_combined_text"
df[combined_col] = df[selected_cols].fillna("").astype(str).agg(" ".join, axis=1)

# Dataset preview (collapsed by default)
with st.expander("Dataset preview (quick)"):
    preview_rows = st.slider("Preview rows", min_value=5, max_value=200, value=10, step=5)
    st.dataframe(df.head(preview_rows))
    if show_preview_all:
        st.write("Showing full dataset:")
        st.dataframe(df)

# Don't proceed until Analyze pressed
if not analyze_btn:
    st.info("Ready. Tap Analyze to run.")
    st.stop()

# Prepare analyzer
det = detect_sentiment_analyzer()
score_fn: Callable[[str], Dict[str, float]] = det["score_fn"]  # type: ignore
analyzer_name = det["name"]

# Compact analyzer notice (single-line)
st.caption(f"Analyzer: {analyzer_name}")

# Optionally sample
if sample_n and 0 < sample_n < len(df):
    df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

# Process with progress
n = len(df)
progress = st.progress(0)
status = st.empty()
clean_texts = []
scores = []
for i, raw in enumerate(df[combined_col].astype(str)):
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

# Side column: quick metrics and download
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
        # show the selected original text columns plus combined and compound
        cols_to_show = [c for c in selected_cols if c in results.columns]
        display_cols = cols_to_show + ["_clean_text", "compound"]
        st.dataframe(sub[display_cols].head(8))
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
    # Show original selected columns, combined cleaned text, label, compound
    cols_preview = [c for c in selected_cols if c in results.columns]
    preview_cols = cols_preview + ["_clean_text", "_label", "compound"]
    st.dataframe(results[preview_cols].head(12))
    st.download_button("Download full results CSV", data=results.to_csv(index=False).encode("utf-8"),
                       file_name="sentiment_results_full.csv")

# Minimal Help (collapsed) - lines removed per request
with st.expander("Help & notes", expanded=False):
    st.write("- Use the top controls to upload data, choose up to three text columns, and run analysis.")
    st.write("- Controls are arranged for mobile and desktop; the layout is wider on desktop for easier reading.")