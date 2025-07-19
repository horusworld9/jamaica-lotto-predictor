import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st

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

        date = cols[0].text.strip()
        try:
            numbers = [int(n.strip()) for n in cols[1:7]]
            bonus = int(cols[7].text.strip())
            data.append({
                'date': pd.to_datetime(date),
                'n1': numbers[0],
                'n2': numbers[1],
                'n3': numbers[2],
                'n4': numbers[3],
                'n5': numbers[4],
                'n6': numbers[5],
                'bonus': bonus
            })
        except:
            continue  # Skip bad rows

    return pd.DataFrame(data)
def build_features(df, window=50):
    last = df.tail(window)
    freq = last[[f'n{i}' for i in range(1, 7)]].stack().value_counts(normalize=True)
    recency = {i: (window - last[::-1][[f'n{j}' for j in range(1, 7)]].apply(lambda col: col.eq(i).idxmax()).index[0]) for i in range(1, 39)}
    feats = [freq.reindex(range(1, 39), fill_value=0).values, np.array([recency[i] for i in range(1, 39)])]
    return np.concatenate(feats)

st.title("🎲 Jamaican Lotto Predictor")
st.caption("This app scrapes past draws and predicts 5 sets of Lotto numbers.")

df = scrape_historical()
features = build_features(df)
probabilities = features[:38] * 0.7 + (1 / (features[38:] + 1)) * 0.3
choices = np.argsort(probabilities)[::-1] + 1

def get_ticket(start):
    return sorted(list(map(int, choices[start:start + 6])))

tickets = [get_ticket(i * 6) for i in range(5)]

st.subheader("🔢 Your Predictions")
for i, t in enumerate(tickets, 1):
    st.write(f"Ticket #{i}: {t}")
