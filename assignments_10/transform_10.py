import os
import json
import pandas as pd

import joblib
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"),os.getenv("SUPABASE_KEY"))

# --- Steps ---

# Read records from weather_raw
response = supabase.table("weather_raw").select("*").execute()
raw_rows = response.data
print(f"Fetched {len(raw_rows)} rows from weather_raw.")

# Step Q1
print(f"\nStep Q1:\n")

# Connect to weather_enriched -- select date column
enriched_response = supabase.table("weather_enriched").select("date").execute()

# Check for processed records
already_done = {row["date"] for row in enriched_response.data}
print(f"{len(already_done)} records processed.\n")

to_classify = [row for row in raw_rows if row["date"] not in already_done]
print(f"Records to classify: {len(to_classify)} (skipping {len(already_done)} already enriched)")

# Step Q2
print(f"\nStep Q2:\n")

# Build the feature DataFrame
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
    
FEATURES = metadata["features"]
df = pd.DataFrame(to_classify)
X = df[FEATURES]

# Predict & predict probabilities
clf = joblib.load("models/weather_classifier.pkl")
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
print("\nHighest confidence (good for running):")
for r in enrichment_records[:3]:
    print(f"  {r['date']}: {r['confidence']:.3f}")

enrichment_records.sort(key=lambda r: abs(r["confidence"] - 0.5))
print("\nMost borderline (closest to 0.5 confidence):")
for r in enrichment_records[:3]:
    print(f"  {r['date']}: {r['confidence']:.3f}")