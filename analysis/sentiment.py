import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Chemins ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Chargement ────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "reviews_clean.csv")
print(f"Lignes chargees : {len(df)}")

# ── Analyse de sentiment VADER ────────────────────────────────────
# VADER est optimise pour les textes courts type avis clients
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text) -> dict:
    if pd.isna(text) or str(text).strip() == "":
        return {"vader_score": None, "vader_label": None}
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"vader_score": round(compound, 4), "vader_label": label}

results = df["full_text"].apply(get_vader_sentiment)
df["vader_score"] = results.apply(lambda x: x["vader_score"])
df["vader_label"] = results.apply(lambda x: x["vader_label"])

# ── Comparaison rating vs sentiment VADER ─────────────────────────
# Les avis ou le texte contredit la note sont les plus interessants
df["mismatch"] = (
    df["sentiment_label"].notna() &
    df["vader_label"].notna() &
    (df["sentiment_label"] != df["vader_label"])
)

# ── Rapport ───────────────────────────────────────────────────────
print(f"\nDistribution VADER :")
print(df["vader_label"].value_counts())

print(f"\nTaux de concordance rating / VADER :")
total_with_text = df["vader_label"].notna().sum()
matches = (df["sentiment_label"] == df["vader_label"]).sum()
print(f"  {matches}/{total_with_text} soit {round(matches/total_with_text*100, 1)}%")

print(f"\nAvis en contradiction (note vs texte) : {df['mismatch'].sum()}")
print(f"\nExemples de contradictions :")
print(
    df[df["mismatch"]][["brand", "rating", "sentiment_label", "vader_label", "full_text"]]
    .head(5)
    .to_string()
)

# ── Sauvegarde ────────────────────────────────────────────────────
df.to_csv(ROOT / "data" / "reviews_sentiment.csv", index=False, encoding="utf-8-sig")
print(f"\nFichier sauvegarde : data/reviews_sentiment.csv")