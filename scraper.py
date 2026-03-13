import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

BRANDS = {
    "The Ordinary": "theordinary.com",
    "CeraVe":       "www.cerave.com",
    "Typology":     "typology.com",
    "Sephora":      "www.sephora.fr",
    "Nuxe":         "nuxe.com",
}

MAX_PAGES = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}


def get_soup(url: str) -> BeautifulSoup | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser")
        else:
            print(f"Status {response.status_code} pour {url}")
            return None
    except requests.RequestException as e:
        print(f"Erreur requete : {e}")
        return None


def clean_text(tag) -> str | None:
    """Supprime les balises font imbriquees et retourne le texte propre."""
    if not tag:
        return None
    # Supprime toutes les balises font sans toucher au texte
    for font in tag.find_all("font"):
        font.unwrap()
    return tag.get_text(strip=True) or None


def parse_reviews(soup: BeautifulSoup, brand: str) -> list[dict]:
    reviews = []
    cards = soup.find_all("article")

    for card in cards:
        try:
            # Rating
            star_img = card.find("img", class_=re.compile("starRating"))
            if star_img and star_img.get("alt"):
                match = re.search(r"(\d)", star_img["alt"])
                rating = int(match.group(1)) if match else None
            else:
                rating = None

            # Titre
            title_tag = card.find("h2", {"data-service-review-title-typography": True})
            title = clean_text(title_tag)

            # Corps
            body_tag = card.find("p", {"data-service-review-text-typography": True})
            body = clean_text(body_tag)

            # Date
            date_tag = card.find("time")
            date = date_tag["datetime"][:10] if date_tag else None

            reviews.append({
                "brand":  brand,
                "rating": rating,
                "title":  title,
                "body":   body,
                "date":   date,
            })

        except Exception as e:
            print(f"Erreur parsing carte : {e}")
            continue

    return reviews


def scrape_brand(brand: str, domain: str, max_pages: int) -> list[dict]:
    all_reviews = []
    print(f"Scraping : {brand}")

    for page in range(1, max_pages + 1):
        url = f"https://www.trustpilot.com/review/{domain}?page={page}"
        print(f"  Page {page}/{max_pages}")

        soup = get_soup(url)
        if not soup:
            break

        reviews = parse_reviews(soup, brand)
        if not reviews:
            print(f"  Plus d'avis a la page {page}, arret.")
            break

        all_reviews.extend(reviews)
        time.sleep(random.uniform(1.5, 3.0))

    print(f"  {len(all_reviews)} avis collectes pour {brand}")
    return all_reviews


def scrape_all_brands(brands: dict, max_pages: int) -> pd.DataFrame:
    all_data = []
    for brand, domain in brands.items():
        reviews = scrape_brand(brand, domain, max_pages)
        all_data.extend(reviews)
    return pd.DataFrame(all_data)


if __name__ == "__main__":
    print("Demarrage du scraping Trustpilot — Skincare Brands\n")

    df = scrape_all_brands(BRANDS, MAX_PAGES)

    output_path = "data/reviews_raw.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nScraping termine.")
    print(f"Total avis collectes : {len(df)}")
    print(f"Fichier sauvegarde   : {output_path}")
    print(f"\nApercu :")
    print(df.head())
    print(f"\nValeurs manquantes :")
    print(df.isnull().sum())