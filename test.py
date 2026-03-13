import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "data" / "reviews_sentiment.csv")

print("=== CHIFFRES POUR LE README ===")
print(f"Total avis : {len(df)}")
print(f"Periode : {df['date'].min()} -> {df['date'].max()}")
print(f"\nMoyenne par marque :")
print(df.groupby("brand")["rating"].mean().round(2).sort_values())
print(f"\n% negatifs par marque :")
neg = df[df["sentiment_label"] == "negative"].groupby("brand").size()
total = df.groupby("brand").size()
print((neg/total*100).round(1).sort_values(ascending=False))
print(f"\nTop 5 mots negatifs les plus frequents :")
from collections import Counter
import re
neg_words = " ".join(df[df["sentiment_label"] == "negative"]["full_text"].dropna())
words = [w for w in re.findall(r'\b\w+\b', neg_words.lower()) if len(w) > 4]
print(Counter(words).most_common(10))