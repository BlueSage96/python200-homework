import os, requests, json, sklearn
from dotenv import load_dotenv
import joblib
import pandas as pd

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
    # fetches 2023 daily weather data
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

# Load_raw Task
@task(retries=2, retry_delay_seconds=5)
def load_raw(records:list) -> None:
    response = (
        # call weather_raw
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_raw.")

# Transform Task
@task
def transform(raw_records: dict) -> list:
    # incremental check: fetches dates already in weather_raw & skips them
    already_done = {
        r["date"]
        for r in supabase.table("weather_enriched")
         .select("date").execute().data
    }
    
    # Records to transform
    to_process = [r for r in raw_records if r["date"] not in already_done]
    print(f"Records to transform: {len(to_process)} (skipping {len(already_done)} already enriched).")
    
    # Records already enriched
    if not to_process:
        print("All records already enriched - nothing to do.")
        return []
    
    # Load weather_classifier
    clf = joblib.load("models/weather_classifier.pkl")
    df = pd.DataFrame(to_process)
    X = df[FEATURES]

    # Runs predict & predict_proba on unprocessed records
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)[:,1]
    print(f"ML classification complete. Good days: {int(predictions.sum())}") / {len(predictions)}
    
    # Creates enrichment records
    enrichment_records = [
        {
            "date": to_process[i]["date"],
            "good_for_running": bool(predictions[i]),
            "confidence": round(float(probabilities[i]), 4),
            "llm_summary": None,
        }
        for i in range(len(to_process))
    ]
    
    # Prediction text based on weather conditions & "good_for_running"
    for i, record in enumerate(enrichment_records):
        raw_row = to_process[i]
        prediction_text = "good for running" if record["good_for_running"] else "not ideal for running"
        user_message = (
            f"Date: {raw_row['date']}\n"
            f"High: {raw_row['temperature_2m_max']}°C\n"
            f"Precipitation: {raw_row['precipitation_sum']}mm\n"
            f"Max wind speed: {raw_row['wind_speed_10m_max']}km/h\n"
            f"Model prediction: {prediction_text} (confidence: {record['confidence']:.0%})"
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=100,
            )
            record["llm_summary"] = response.choices[0].message.content.strip() or "Recommendation unavailable."

        # Handles LLM errors with a fallback string
        except Exception as e:
            print(f"    LLM error on {record['date']}: {e}")
            record["llm_summart"] = "Recommendation unavailable."
            
        # Prints progress every 50 seconds
        if (i + 1) % 50 == 0:
            print(f"    LLM enriched {i + 1} / {len(enrichment_records)} records.")
    print(f"Transform complete: {len(enrichment_records)} records enriched.")
    # Returns complete list of enrichment records
    return enrichment_records
        
@task
def load_enriched(records: list) -> None:
    ...

@flow(log_prints=True)
def etl_pipeline():
    # calls Open-Meteo historical archive API
    raw_records = extract("https://archive-api.open-meteo.com/v1/archive")
    load_raw(raw_records)
    enrichment_records = transform(raw_records)
    load_enriched(enrichment_records)

if __name__ == "__main__":
    etl_pipeline()