import os
import pandas as pd
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
print(f"\nStep 01:\n")

# Connect to weather_enriched -- select date column
enriched_response = supabase.table("weather_enriched").select("date").execute()

# Check for processed records
already_done = {row["date"] for row in enriched_response.data}
print(f"{len(already_done)} records processed.\n")

to_classify = [row for row in raw_rows if row["date"] not in already_done]
print(f"Records to classify: {len(to_classify)} (skipping {len(already_done)} already enriched)\n")