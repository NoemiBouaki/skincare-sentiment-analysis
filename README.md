# Skincare Brands — Customer Sentiment Analysis

Sentiment analysis of 859 Trustpilot reviews collected across 5 skincare brands,
combining web scraping, NLP, and interactive data visualization.

**Live dashboard → [Insert Streamlit link here]**

---

## The Business Question

Star ratings tell you *that* customers are unhappy. They rarely tell you *why*.

This project goes further: by analyzing the text of reviews alongside ratings,
it surfaces recurring complaints, tracks sentiment trends over time, and flags
cases where customer language contradicts their own rating — a signal that
traditional metrics miss entirely.

---

## Key Findings

- **Sephora and The Ordinary have the highest dissatisfaction rates** — 71.4% and
  66.8% of their reviews are negative, driven overwhelmingly by delivery and
  customer service failures, not product quality.

- **The most frequent words in negative reviews** are *order*, *customer*, and
  *service* — appearing in 299, 284, and 260 reviews respectively. Product
  complaints are secondary.

- **Typology and Nuxe significantly outperform** the group, with negative review
  rates of 23% and 24.6% and average ratings of 3.86 and 3.92.

- **VADER sentiment analysis agrees with star ratings 75.4% of the time.**
  The remaining 24.6% reveals nuanced cases — customers who write positively
  despite low ratings (mixed experiences) or express frustration within
  otherwise positive reviews.

- **Data covers 2014 to March 2026**, allowing trend analysis across a decade
  of customer feedback.

---

## Project Structure
```
skincare-sentiment-analysis/
│
├── data/
│   ├── reviews_raw.csv
│   ├── reviews_clean.csv
│   └── reviews_sentiment.csv
│
├── analysis/
│   ├── clean.py
│   └── sentiment.py
│
├── dashboard/
│   └── app.py
│
├── scraper.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Stack

| Layer | Tools |
|---|---|
| Scraping | Python, Requests, BeautifulSoup |
| Data processing | Pandas |
| Sentiment analysis | VADER (vaderSentiment) |
| Visualization | Streamlit, Plotly, WordCloud |

---

## Methodology

**Scraping** — Reviews were collected from Trustpilot across 5 brands
(The Ordinary, CeraVe, Typology, Sephora, Nuxe) using Requests and
BeautifulSoup. Polite scraping practices were applied: randomized delays
between requests, no authentication bypass.

**Sentiment labeling** — Two sentiment signals are computed per review:
- *Rating-based label*: derived directly from the star rating
  (1–2 → negative, 3 → neutral, 4–5 → positive)
- *VADER label*: derived from the review text using a lexicon-based model

Comparing both signals surfaces the most analytically interesting cases.

**Limitations** — VADER is optimized for English. Reviews written in French
(primarily Sephora and Nuxe) produce less reliable scores. A multilingual
model such as CamemBERT would improve accuracy on French-language data.

---

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Collect reviews
python scraper/scraper.py

# Clean data
python analysis/clean.py

# Run sentiment analysis
python analysis/sentiment.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## About

Built as a portfolio project demonstrating end-to-end data analysis skills:
data collection, cleaning, NLP, and interactive visualization.

Open to freelance data analysis and NLP projects.
→ [Your Upwork profile link here]