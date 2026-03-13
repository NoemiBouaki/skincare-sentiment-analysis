import pandas as pd
import re
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Chargement ────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "reviews_raw.csv")
print(f"Lignes brutes : {len(df)}")

# ── Suppression des doublons ──────────────────────────────────────
df = df.drop_duplicates(subset=["brand", "title", "body", "date"])
print(f"Apres deduplication : {len(df)}")

# ── Typage des colonnes ───────────────────────────────────────────
df["date"]   = pd.to_datetime(df["date"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")

# ── Nettoyage du texte ────────────────────────────────────────────
def clean_text(text: str) -> str | None:
    if pd.isna(text):
        return None
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip() or None

df["title"] = df["title"].apply(clean_text)
df["body"]  = df["body"].apply(clean_text)

# ── Colonne texte complet ─────────────────────────────────────────
df["full_text"] = df.apply(
    lambda row: " ".join([
        str(x) for x in [row["title"], row["body"]]
        if pd.notna(x) and str(x).strip() != ""
    ]),
    axis=1
)
df["full_text"] = df["full_text"].replace("", None)

# ── Sentiment base sur le rating ──────────────────────────────────
def rating_to_sentiment(rating) -> str | None:
    if pd.isna(rating):
        return None
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"

df["sentiment_label"] = df["rating"].apply(rating_to_sentiment)

# ── Colonnes date ─────────────────────────────────────────────────
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

# ── Rapport ───────────────────────────────────────────────────────
print(f"\nShape final : {df.shape}")
print(f"\nValeurs manquantes :")
print(df.isnull().sum())
print(f"\nDistribution des sentiments :")
print(df["sentiment_label"].value_counts())
print(f"\nDistribution par marque :")
print(df["brand"].value_counts())
print(f"\nApercu :")
print(df.head())

# ── Sauvegarde ────────────────────────────────────────────────────
df.to_csv(ROOT / "data" / "reviews_clean.csv", index=False, encoding="utf-8-sig")
print(f"\nFichier sauvegarde : data/reviews_clean.csv")