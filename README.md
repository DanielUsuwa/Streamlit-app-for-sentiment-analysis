# Sentiment Analysis Web Application (Streamlit)

🔗 **Live Application**  
https://daniel-usuwa-app-for-sentiment-analysis-gxnzerdue47s2psmu9wcqf.streamlit.app/

---

## Project Overview
This project is an end-to-end sentiment analysis web application that I designed, built, and deployed using Python and Streamlit.  
The application transforms raw textual data (such as customer reviews or feedback) into structured sentiment insights through a complete NLP and visualization pipeline.

Rather than remaining at a notebook or proof-of-concept stage, the focus of this project is usability, robustness, and real-world deployment, allowing both technical and non-technical users to analyze sentiment directly from uploaded datasets.

---

## Business Context & Use Case
Understanding customer sentiment is critical for decision-making across marketing, brand management, and customer experience functions.  
This application can be used to:

- Analyze customer reviews, survey responses, or written feedback
- Support brand perception and campaign evaluation
- Identify dominant themes and sentiment drivers through visual text exploration
- Provide fast qualitative insights without requiring complex analytics tools or coding skills

---

## Solution Approach
I designed and implemented an end-to-end sentiment analysis pipeline that operates directly on real-world datasets rather than static notebooks.

The application:
- Accepts CSV uploads and supports analysis across up to three text columns simultaneously
- Cleans and preprocesses text data (HTML removal, URL stripping, normalization, optional number removal)
- Automatically detects and applies the best available sentiment analyzer (NLTK VADER, vaderSentiment, or a keyword-based fallback)
- Classifies sentiment into positive, neutral, and negative categories using compound polarity scores

To support exploratory text analysis and insight discovery, the app incorporates **word cloud visualisation**:
- Image-based WordClouds are generated when the library is available
- A robust visual fallback cloud (frequency-based using Altair) is used when WordCloud is unavailable  
This ensures consistent insight delivery across different environments and deployments.

Results are presented through:
- Interactive sentiment distribution charts
- Per-sentiment breakdowns with top-word frequency analysis
- Downloadable CSV outputs for further analysis or reporting

The interface is built with Streamlit and includes dataset previews, sampling controls, and a light/dark theme toggle, making the tool accessible to both technical and non-technical users.

---

## Key Features
- CSV-based sentiment analysis (single or multi-column text input)
- Automatic sentiment analyzer detection with fallback logic
- Word cloud generation with environment-safe fallback
- Interactive visualizations (distribution charts, top-word analysis)
- Dataset preview and sampling controls
- Light / Dark theme toggle for improved usability
- Downloadable, analysis-ready result files
- Deployed as a live web application

---

## Tech Stack
- Python (Pandas, NumPy)
- Natural Language Processing (text preprocessing, sentiment scoring)
- Sentiment Analysis (NLTK VADER / vaderSentiment with fallback logic)
- Text Visualisation (WordCloud, Altair-based fallback cloud)
- Data Visualisation (Altair, Matplotlib)
- Web App Framework (Streamlit)

---

## What This Project Demonstrates
- Ability to translate NLP techniques into business-ready tools
- Experience building robust, environment-aware analytics applications
- Practical understanding of deployment constraints and fallback strategies
- Strong focus on insight communication, not just model output
- End-to-end ownership: data ingestion → processing → visualization → deployment

---


## Author
**Daniel Usuwa**  
Business & Data Analyst  
MSc Digital Marketing & Data Analytics – Grenoble Ecole de Management (Paris)
