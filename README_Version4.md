# Sentiment Analyzer 

This Streamlit app performs sentiment analysis on a CSV of text reviews. It will:
- detect and use the best available sentiment analyzer (NLTK VADER if available, then vaderSentiment, otherwise a lightweight fallback),
- generate a Word Cloud of positive reviews if the `wordcloud` package is installed in the environment,
- provide a download of scored results (CSV) and a sentiment distribution chart.

Important notes about WordCloud and dependencies
- WordCloud is optional in the app. The code safely guards the import so the app will start even when `wordcloud` isn't installed.
- To enable WordCloud on Streamlit Cloud (or other hosts), add `wordcloud` and `pillow` to `requirements.txt` and redeploy.
  - Be aware: `wordcloud` sometimes requires compilation; on some hosts / Python versions installing it may fail. If installation fails, you can run the app locally with these packages installed.
- For higher-quality sentiment results, add `nltk` and ensure the `vader_lexicon` resource is present (or add `vaderSentiment`) in `requirements.txt`. The app will automatically switch to VADER when available.

Quick start (local)
1. Create a virtual environment and activate it:
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows

2. Install requirements (edit `requirements.txt` to uncomment packages you want):
   pip install -r requirements.txt

3. Run:
   streamlit run app.py

Quick start (Streamlit Cloud)
- Push this repo to GitHub, make sure `app.py` is the main module, and add `requirements.txt` (with `wordcloud` uncommented only if your host supports building it).
- Deploy via Streamlit Cloud. If WordCloud fails to install on deploy, the app will still run; WordCloud UI will be disabled and you'll see instructions in-app.

UX & Error handling
- The app informs which sentiment analyzer it's using.
- If WordCloud is unavailable, the app continues and shows how to enable WordCloud.
- If your dataset is large, use the Sample size option to analyze a subset for quick iterations.

If you want, I can:
- Try to pin specific versions of `wordcloud`/`pillow` that are known to have wheels for your host Python version, or
- Prepare a branch that removes optional packages and only includes production-safe features.
