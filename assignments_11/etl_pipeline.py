import os, requests, json, sklearn
from dotenv import load_dotenv
import numpy as np
from prefect import flow, task
from supabase import create_client
from openai import OpenAI

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

# Extract Task
latitude = 34.99713980841658
longitude = -78.33071903597848

@task(retries=2,retry_delay_seconds=10)
def extract(url: str) -> list:
    # calls Open-Meteo & fetches 2023 daily weather data
    url = "https://archive-api.open-meteo.com/v1/archive" 
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": [FEATURES],
        "timezone": "America/New_York",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    # Converts columnar API response into a list of dictionaries
    daily = response.json()["daily"]
    records = [
        {
            "date": daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }
        for i in range(len(daily["time"]))
    ]
    
    print(f"Extracted {len(records)} daily records fron Open-Mateo")
    return records

# load_raw task
@task(retries=2, retry_delay_seconds=5)
def load_raw(records:list) -> None:
    response = (
        # call weather_raw
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_raw.")

@task
def transform(data: dict) -> list:
    ...

@task
def load(records: list) -> None:
    ...

@flow(log_prints=True)
def etl_pipeline():
    data    = extract("https://archive-api.open-meteo.com/v1/archive")
    records = transform(data)
    load(records)

if __name__ == "__main__":
    etl_pipeline()