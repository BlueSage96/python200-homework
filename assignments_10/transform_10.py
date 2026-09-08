import os
import json
import pandas as pd

import joblib
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

# Robust wherever Python launches
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"),os.getenv("SUPABASE_KEY"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- Steps ---

# Read records from weather_raw
response = supabase.table("weather_raw").select("*").execute()
raw_rows = response.data

with open(os.path.join(MODEL_DIR, "weather_classifier_metadata.json")) as f:
    metadata = json.load(f)
        

# Step Q1
print(f"\nStep Q1:\n")

# Connect to weather_enriched -- select date column
enriched_response = supabase.table("weather_enriched").select("date").execute()

# Check for processed records
already_done = {row["date"] for row in enriched_response.data}

to_classify = [row for row in raw_rows if row["date"] not in already_done]

print(
    f"Raw records: {len(raw_rows)} | "
    f"Already enriched: {len(already_done)} | "
    f"Processing this run: {len(to_classify)}"
)

if to_classify:
    # Step Q2
    print(f"\nStep Q2:\n")

    # Build the feature DataFrame
    FEATURES = metadata["features"]
    df = pd.DataFrame(to_classify)
    X = df[FEATURES]

    # Predict & predict probabilities
    clf = joblib.load(os.path.join(MODEL_DIR, "weather_classifier.pkl"))
    predictions = clf.predict(X) #array of 0s & 1s
    probabilities = clf.predict_proba(X)[:,1] # Probability of class 1 (good for running])

    print(f"\nGood days predicted: {predictions.sum()} / {len(predictions)}")
    print(f"\nConfidence range: {probabilities.min():.2f} - {probabilities.max():.2f}\n")

    # Number of good days are 138 and the confidence range of 0.0-0.98.

    # Build enrichment records -- combine the predictions back with the date for each row
    enrichment_records = []
    for i, row in enumerate(to_classify):
        enrichment_records.append({
            "date": row["date"],
            "good_for_running": bool(predictions[i]),
            "confidence": round(float(probabilities[i]),4),
        })
        
    # Sanity check before adding LLM
    good_days = [r for r in enrichment_records if r["good_for_running"]]
    skip_days = [r for r in enrichment_records if not r["good_for_running"]]

    print(f"Good days: {len(good_days)} ({len(good_days)/len(enrichment_records):.0%})")
    print(f"Skip days: {len(skip_days)}")

    # Show a few high-confidence and borderline predictions
    enrichment_records.sort(key=lambda r: r["confidence"], reverse=True)
    print("\nHighest confidence (good for running):\n")
    for r in enrichment_records[:3]:
        print(f"  {r['date']}: {r['confidence']:.3f}")

    enrichment_records.sort(key=lambda r: abs(r["confidence"] - 0.5))
    print("\nMost borderline (closest to 0.5 confidence):\n")
    for r in enrichment_records[:3]:
        print(f"  {r['date']}: {r['confidence']:.3f}")
        
    # Step Q3

    print(f"\nStep Q3:\n")

    # Prompt
    SYSTEM_PROMPT = (
        "You are writing a one-sentence running recommendation for a daily weather summary app. "
        "You will receive weather conditions for a single day and a machine learning prediction "
        "about whether the day is good for running. "
        "Write exactly one sentence — direct, practical, and specific to the conditions. "
        "Do not use bullet points, headers, or phrases like 'Based on the data'."
    )

    def make_user_message(row, good_for_running, confidence):
        prediction_text = "good for running" if good_for_running else "not ideal for running"
        return (
            f"Date: {row['date']}\n"
            f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
            f"Precipitation: {row['precipitation_sum']} mm\n"
            f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
        )

    # raw_rows is the original list of dicts from weather_raw
    # enrichment_records is the list from the ML step, indexed in the same order

    for i, record in enumerate(enrichment_records):
        raw_row = next(r for r in to_classify if r["date"] == record["date"])
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": make_user_message(
                            raw_row,
                            record["good_for_running"],
                            record["confidence"]
                        )
                    }
                ],
                max_tokens=100
            )
            summary = response.choices[0].message.content.strip()
            record["llm_summary"] = summary
        
        except Exception as e:
            print(f"API error on {record['date']}: {e}")
            record["llm_summary"] = "Recommendation unavailable."
            
        # Fallback string & progress print every 50 records
        if (i + 1) % 50 == 0 or (i + 1) == len(enrichment_records):
            print(f"\nEnriched {i + 1} / {len(enrichment_records)} records...")

    # Step Q4
    print(f"\nStep Q4:\n")

    # upsert to weather_enriched once records have an llm_summary
    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_enriched")
else:
    print("Nothing to do - all records already enriched")
    
# Step 05
print(f"\nStep Q5:\n")

# Confirm weather_enriched results
check = supabase.table("weather_enriched").select("*").limit(5).execute()
for row in check.data:
    print(
        f"{row['date']} | "
        f"good={row['good_for_running']} | "
        f"conf={row['confidence']:.2f} | "
        f"summary={row['llm_summary']}"
    )
    
#count how many records were classified as good
good_count = (
    supabase.table("weather_enriched")
    .select("date", count="exact")
    .eq("good_for_running", True)
    .execute()
)

total_count = (
    supabase.table("weather_enriched")
    .select("date", count="exact")
    .execute()
)

print(f"Total good-for-running days: {good_count.count}")
print(f"Total rows in weather_enriched: {total_count.count}")

# Step Q6

# The classifier was trained using weather data from my hometown, so I would be
# careful about assuming that it will generalize equally well to cities with very
# different climates. A model trained on one location may perform less reliably
# when weather patterns differ significantly from the training data. In this
# pipeline, the LLM does not override the classifier because good_for_running is
# set by the machine learning model, while the LLM only adds a natural-language
# summary. This makes the LLM transform additive rather than authoritative. If
# the pipeline processed 50,000 records, my main concern would be the cost and
# time required to make that many LLM API calls. Incremental processing would
# help by preventing previously enriched records from being unnecessarily sent
# through the classifier and LLM again.