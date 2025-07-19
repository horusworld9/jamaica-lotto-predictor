import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Scrape past lotto results
def scrape_historical():
    url = 'https://www.jamaicaindex.com/lottery/results/lotto'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find("table")
    rows = table.find_all("tr")[1:]  # skip header

    data = []
    for row in rows:
        cols = row.find_all("td")
        if not cols or len(cols) < 8:
            continue
        try:
            date = pd.to_datetime(cols[0].text.strip())
            numbers = [int(col.text.strip()) for col in cols[1:7]]
            bonus = int(cols[7].text.strip())
            data.append({
                'date': date,
                'n1': numbers[0],
                'n2': numbers[1],
                'n3': numbers[2],
                'n4': numbers[3],
                'n5': numbers[4],
                'n6': numbers[5],
                'bonus': bonus
            })
        except ValueError:
            continue  # Skip malformed rows
    return pd.DataFrame(data)

# Feature engineering
def build_features(df, window=50):
    last = df.tail(window)
    freq = last[[f'n{i}' for i in range(1, 7)]].stack().value_counts(normalize=True)
    recency = {}
    for i in range(1, 39):
        recent = last[::-1][[f'n{j}' for j in range(1, 7)]]
        found = recent.apply(lambda col: col.eq(i).idxmax() if i in col.values else None).dropna()
        if not found.empty:
            recency[i] = window - found.min()
        else:
            recency[i] = window
    feats = [freq.reindex(range(1, 39), fill_value=0).values, np.array([recency[i] for i in range(1, 39)])]
    return np.concatenate(feats)

# Generate 5 predictions
def generate_predictions(df):
    features = build_features(df)
    probabilities = features[:38] * 0.7 + (1 / (features[38:] + 1)) * 0.3
    choices = np.argsort(probabilities)[::-1] + 1

    def get_ticket(start):
        return sorted(list(map(int, choices[start:start + 6])))

    return [get_ticket(i * 6) for i in range(5)]

# Streamlit UI
st.set_page_config(page_title="Jamaican Lotto Predictor 🎲", page_icon="🎯")
st.title("🎲 Jamaican Lotto Predictor")
st.caption("Pulls real draw data and gives you 5 predicted number sets based on frequency + recency.")

with st.spinner("Scraping past draw data..."):
    df = scrape_historical()

if df.empty:
    st.error("Could not load draw data. Try again later.")
else:
    predictions = generate_predictions(df)

    st.subheader("🔢 Your Lucky Picks")
    for i, p in enumerate(predictions, 1):
        st.write(f"**Ticket #{i}**: `{p}`")

    st.success("Prediction complete based on last 50 draws.")
