import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import random

st.set_page_config(page_title="Jamaica Lotto Predictor", layout="centered")

def scrape_historical():
    url = 'https://www.jamaicaindex.com/lottery/results/lotto'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/113.0.0.0 Safari/537.36'
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            st.error("Failed to fetch draw data.")
            return pd.DataFrame()

        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("Draw table not found.")
            return pd.DataFrame()

        rows = table.find_all("tr")[1:]
        data = []

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 8:
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
                continue

        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def generate_predictions(freq_weights=None):
    pool = list(range(1, 41))
    if freq_weights:
        weights = [freq_weights.get(i, 0.01) for i in pool]
        return random.choices(pool, weights=weights, k=6)
    else:
        return random.sample(pool, 6)

st.title("ð¯ð² Jamaica Lotto Predictor")
st.markdown("**Predict numbers based on recent draw trends.**")

df = scrape_historical()
if df.empty:
    st.stop()

st.subheader("ð Recent Draws")
st.dataframe(df.head(10))

# Frequency-based weighting
recent = df.tail(30)
freq = recent[['n1','n2','n3','n4','n5','n6']].stack().value_counts(normalize=True)
freq_weights = freq.to_dict()

st.subheader("ð® Your Predictions")
for i in range(5):
    prediction = generate_predictions(freq_weights)
    st.write(f"Set {i+1}: ", sorted(prediction))
