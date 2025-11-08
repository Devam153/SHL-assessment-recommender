
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

def scrape_assessment_durations(csv_path, output_path=None):
    """
    Scrape the approximate completion time from SHL assessment pages
    and update the CSV file with the duration information.
    Args:
        csv_path: Path to the input CSV file with assessment links
        output_path: Path to save the output CSV file (if None, will use default name)
    
    Returns:
        DataFrame with updated duration information
    """

    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if "Duration" not in df.columns:
        df["Duration"] = None
    
    _minutes_re = re.compile(r"(\d+)(?:\-(\d+))?\s*minute", re.IGNORECASE)
    _time_in_minutes_re = re.compile(r"completion\s+time\s+in\s+minutes\s*[=:]\s*(\d+)", re.IGNORECASE)
    
    print("Starting to scrape duration information...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if pd.notna(row["Duration"]) and str(row["Duration"]).strip():
            continue
            
        link_field = None
        if "Link" in row:
            link_field = "Link"
        
        if not link_field:
            print(f"[!] Error at row {idx}: No link column found. Available columns: {row.index.tolist()}")
            continue
            
        url = row[link_field]
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SHL-Scraper/1.0)"},
                timeout=10
            )
            
            resp.encoding = 'utf-8'
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            duration_text = ""
            
            assessment_length = soup.find(string=re.compile("Assessment length", re.I))
            if assessment_length and assessment_length.parent:
                parent_element = assessment_length.parent
                for sibling in parent_element.next_siblings:
                    if hasattr(sibling, 'get_text'):
                        text = sibling.get_text()
                        if "completion time" in text.lower() or "minutes" in text.lower():
                            duration_text = text
                            break
                
                if not duration_text and parent_element.parent:
                    for element in parent_element.parent.find_all(text=True):
                        text = element.strip()
                        if "completion time" in text.lower() or "minutes" in text.lower():
                            duration_text = text
                            break
            
            if not duration_text:
                dt = soup.find("dt", string=re.compile("Approximate Completion Time", re.I))
                if dt and dt.find_next_sibling("dd"):
                    duration_text = dt.find_next_sibling("dd").get_text()
            
            if not duration_text:
                p = soup.find(string=re.compile("Approximate completion time", re.I))
                if p:
                    duration_text = p
                    if p.parent:
                        duration_text = p.parent.get_text()
            
            if not duration_text:
                elements = soup.find_all(string=re.compile(r"minutes\s*[=:]\s*\d+", re.I))
                if elements:
                    duration_text = elements[0]
            
           
            minutes = None
            m = _time_in_minutes_re.search(duration_text)
            if m:
                minutes = f"{m.group(1)} min"
            else:
                m = _minutes_re.search(duration_text)
                if m:
                    if m.group(2):  
                        minutes = f"{m.group(1)}-{m.group(2)} min"
                    else:
                        minutes = f"{m.group(1)} min"
            '''Use two regexes to extract minutes:

_time_in_minutes_re: completion\s+time\s+in\s+minutes\s*[=:]\s*(\d+) (captures explicit “completion time in minutes: 20”)

_minutes_re: (\d+)(?:\-(\d+))?\s*minute (captures 15 minute or 10-15 minutes)
'''
            if minutes:
                df.at[idx, "Duration"] = minutes
                print(f"Found duration for row {idx}: {minutes}")
            else:
                print(f"[!] Could not find duration for row {idx} - Text found: {duration_text}")
                
        except Exception as e:
            print(f"[!] Error at row {idx} ({url}): {str(e)}")
        
    

    if output_path is None:
        output_path = "src/data/shl_full_catalog_with_duration.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Done! Saved to {output_path}")
    
    return df

if __name__ == "__main__":

    scrape_assessment_durations(
        csv_path="src/data/shl_full_catalog.csv",
        output_path="src/data/shl_full_catalog_with_duration.csv"
    )
