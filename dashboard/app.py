import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Skincare Sentiment Analysis",
    page_icon=None,
    layout="wide"
)

# ── Chargement ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "reviews_sentiment.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.title("Filtres")

brands = ["All"] + sorted(df["brand"].unique().tolist())
selected_brand = st.sidebar.selectbox("Marque", brands)

sentiments = ["All", "positive", "neutral", "negative"]
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)

# ── Filtrage ──────────────────────────────────────────────────────
filtered = df.copy()
if selected_brand != "All":
    filtered = filtered[filtered["brand"] == selected_brand]
if selected_sentiment != "All":
    filtered = filtered[filtered["sentiment_label"] == selected_sentiment]

# ── Header ────────────────────────────────────────────────────────
st.title("Skincare Brands — Customer Sentiment Analysis")
st.markdown("Analysis of Trustpilot reviews across 5 skincare brands")

# ── KPIs ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", len(filtered))
col2.metric(
    "Average Rating",
    round(filtered["rating"].mean(), 2) if not filtered.empty else "—"
)
col3.metric(
    "Positive Reviews",
    f"{round(filtered[filtered['sentiment_label'] == 'positive'].shape[0] / len(filtered) * 100, 1)}%"
    if not filtered.empty else "—"
)
col4.metric(
    "VADER Concordance",
    f"{round((filtered['sentiment_label'] == filtered['vader_label']).sum() / filtered['vader_label'].notna().sum() * 100, 1)}%"
    if filtered['vader_label'].notna().sum() > 0 else "—"
)

st.divider()

# ── Ligne 1 : Distribution ratings + Sentiment par marque ─────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Rating Distribution")
    fig = px.histogram(
        filtered,
        x="rating",
        color="brand" if selected_brand == "All" else None,
        nbins=5,
        barmode="overlay",
        labels={"rating": "Rating", "count": "Reviews"},
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Sentiment by Brand")
    sentiment_counts = (
        filtered.groupby(["brand", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        sentiment_counts,
        x="brand",
        y="count",
        color="sentiment_label",
        barmode="group",
        color_discrete_map={
            "positive": "#2ecc71",
            "neutral":  "#f39c12",
            "negative": "#e74c3c",
        },
        labels={"count": "Reviews", "brand": "Brand", "sentiment_label": "Sentiment"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Ligne 2 : Evolution temporelle + Wordcloud ────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Over Time")
    timeline = (
        filtered.groupby(["year", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )
    timeline["year"] = timeline["year"].astype(str)
    fig = px.line(
        timeline,
        x="year",
        y="count",
        color="sentiment_label",
        markers=True,
        color_discrete_map={
            "positive": "#2ecc71",
            "neutral":  "#f39c12",
            "negative": "#e74c3c",
        },
        labels={"count": "Reviews", "year": "Year"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Most Frequent Words")
    text_data = filtered["full_text"].dropna()

    if len(text_data) > 0:
        tab_pos, tab_neg = st.tabs(["Positive reviews", "Negative reviews"])

        with tab_pos:
            pos_text = " ".join(
                filtered[filtered["sentiment_label"] == "positive"]["full_text"].dropna()
            )
            if pos_text.strip():
                wc = WordCloud(
                    width=600, height=300,
                    background_color="white",
                    colormap="Greens",
                    max_words=80
                ).generate(pos_text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        with tab_neg:
            neg_text = " ".join(
                filtered[filtered["sentiment_label"] == "negative"]["full_text"].dropna()
            )
            if neg_text.strip():
                wc = WordCloud(
                    width=600, height=300,
                    background_color="white",
                    colormap="Reds",
                    max_words=80
                ).generate(neg_text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
    else:
        st.info("No text data available for the current selection.")

st.divider()

# ── Ligne 3 : Contradictions + Table ─────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Rating vs VADER Mismatch")
    mismatch_counts = pd.DataFrame({
        "Match": ["Concordant", "Contradictory"],
        "Count": [
            (filtered["sentiment_label"] == filtered["vader_label"]).sum(),
            (filtered["sentiment_label"] != filtered["vader_label"]).sum(),
        ]
    })
    fig = px.pie(
        mismatch_counts,
        names="Match",
        values="Count",
        color="Match",
        color_discrete_map={
            "Concordant":   "#2ecc71",
            "Contradictory": "#e74c3c",
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Review Explorer")
    st.dataframe(
        filtered[["brand", "rating", "sentiment_label", "vader_label", "full_text", "date"]]
        .sort_values("date", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=300,
    )