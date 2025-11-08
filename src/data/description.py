import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


def scrape_descriptions(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Scrape the detailed description text from each SHL assessment page and
    add it as a new column to the catalog CSV.

    Args:
        csv_path: Path to input CSV containing at least a 'Link' column.
        output_path: Path to save enriched CSV. If None, appends '_with_desc.csv'.

    Returns:
        DataFrame with an added 'Description' column.
    """
    df = pd.read_csv(csv_path)

    if 'Description' not in df.columns:
        df['Description'] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Scraping descriptions'):
        url = row.get('Link') or row.get('link')
        if not url:
            continue

        if pd.notna(row['Description']) and str(row['Description']).strip():
            continue

        try:
            resp = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; SHL-Desc-Scraper/1.0)'},
                timeout=10
            )
            resp.encoding = 'utf-8'
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')

            desc_text = ''
            h4 = soup.find('h4', string=re.compile(r'Description', re.I))
            if h4:
                p_desc = h4.find_next_sibling('p')
                if p_desc:
                    desc_text = p_desc.get_text(' ', strip=True)

            df.at[idx, 'Description'] = desc_text or None

        except Exception as e:
            print(f"[!] Error scraping row {idx} ({url}): {e}")


    if not output_path:
        if csv_path.lower().endswith('.csv'):
            output_path = csv_path[:-4] + '_with_desc.csv'
        else:
            output_path = csv_path + '_with_desc.csv'

    df.to_csv(output_path, index=False)
    print(f"Descriptions saved to {output_path}")

    return df

if __name__ == '__main__':
    scrape_descriptions(
        csv_path='src/data/shl_full_catalog_with_duration.csv',
        output_path='src/data/shl_full_catalog_with_duration_desc.csv'
    )