import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify

app = Flask(__name__)

def scrape_historical():
    url = 'https://www.jamaicaindex.com/lottery/results/lotto'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    rows = soup.select('table tr')[1:]
    data = []
    for row in rows:
        cols = row.find_all('td')
        if not cols or len(cols) < 8:
            continue
        date = cols[0].text.strip()
        nums = [int(n.text) for n in cols[1:7]]
        bonus = int(cols[7].text)
        data.append({'date': pd.to_datetime(date), **{f'n{i+1}': nums[i] for i in range(6)}, 'bonus': bonus})
    return pd.DataFrame(data)

def build_features(df, window=50):
    last = df.tail(window)
    freq = last[[f'n{i}' for i in range(1, 7)]].stack().value_counts(normalize=True)
    recency = {i: (window - last[::-1][[f'n{j}' for j in range(1, 7)]].apply(lambda col: col.eq(i).idxmax()).index[0]) for i in range(1, 39)}
    feats = [freq.reindex(range(1, 39), fill_value=0).values, np.array([recency[i] for i in range(1, 39)])]
    return np.concatenate(feats)

@app.route("/predict", methods=["GET"])
def predict():
    df = scrape_historical()
    features = build_features(df)
    probabilities = features[:38] * 0.7 + (1 / (features[38:] + 1)) * 0.3
    choices = np.argsort(probabilities)[::-1] + 1

    def get_ticket(start):
        return sorted(list(map(int, choices[start:start + 6])))

    tickets = [get_ticket(i * 6) for i in range(5)]
    return jsonify({"predictions": tickets})

if __name__ == "__main__":
app.run(host="0.0.0.0", port=8080)
