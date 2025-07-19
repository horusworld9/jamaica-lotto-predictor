import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_historical():
    url = 'https://www.jamaicaindex.com/lottery/results/lotto'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/113.0.0.0 Safari/537.36'
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"HTTP status: {r.status_code}")
        if r.status_code != 200:
            return pd.DataFrame()

        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find("table")
        if not table:
            print("No table found on the page.")
            return pd.DataFrame()

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
            except ValueError as ve:
                print(f"Skipping row due to value error: {ve}")
                continue

        df = pd.DataFrame(data)
        print(f"Scraped {len(df)} draws.")
        return df

    except Exception as e:
        print(f"Error during scraping: {e}")
        return pd.DataFrame()
