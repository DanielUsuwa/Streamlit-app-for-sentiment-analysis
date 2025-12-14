# Simple Sentiment Analyzer (Streamlit)

A small, user-friendly Streamlit app for sentiment analysis of text reviews using NLTK + VADER.
It includes preprocessing (HTML/URL removal, punctuation/number removal, tokenization, stopword removal,
and POS-aware lemmatization), VADER sentiment scoring, a sentiment distribution chart, and an optional
word cloud of positive reviews. You can upload a CSV and download results with sentiment scores.

## Files included
- `app.py` — Streamlit application (main).
- `SAMPLE_TEXT.txt` — small sample CSV content and quick test instructions.
- `.gitignore` — recommended ignores for a Python/Streamlit project.

## Requirements
Recommended to create a virtual environment.

Minimum dependencies:
- Python 3.8+
- streamlit
- pandas
- nltk
- wordcloud
- pillow
- plotly

Install with pip:
```bash
pip install streamlit pandas nltk wordcloud pillow plotly
```

Optionally create a `requirements.txt` using:
```bash
pip freeze > requirements.txt
```

## First-time setup
The app will automatically download a few NLTK resources on first run:
- punkt
- stopwords
- wordnet
- omw-1.4
- averaged_perceptron_tagger
- vader_lexicon

This requires internet access. If running on an air-gapped server, download these beforehand:
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
```

## Run the app
```bash
streamlit run app.py
```

Then open the URL provided by Streamlit (usually http://localhost:8501).

## How to use
1. Upload a CSV with a text column (e.g. `review_content`) or click "Use sample data".
2. If the default column is different, type your text column name in the sidebar.
3. Optionally limit the sample size for faster iteration.
4. Toggle wordcloud and preprocessing options as desired.
5. Click "Analyze".
6. View charts, sample rows, and download the results CSV or wordcloud PNG.

## UX & Error handling notes
- The app shows spinners/progress while downloading resources or scoring so users know what's happening.
- Friendly error messages are shown on CSV read errors or missing columns.
- If you experience long delays on the first run: NLTK downloads are most common cause.
- If the CSV is large, consider sampling before running or run the analysis locally in a more robust environment.

## Troubleshooting
- "NLTK resource not found" or similar: Allow the app to download resources or pre-download them using the command above.
- "ModuleNotFoundError" on dependencies: Install packages listed in Requirements.
- If Streamlit serves a stale page: stop and restart the app process.

## Next improvements (suggestions)
- Add authentication or a small API to process files programmatically.
- Add caching for processed datasets to speed up repeated analysis.
- Add a supervised classifier (fine-tune a transformer) for domain-specific better accuracy.

## License
MIT or whatever license you prefer for your project. This repo contains example code and is provided as-is.