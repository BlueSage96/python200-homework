import requests
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Step 01: Extract
print(f"\nStep 01:\n")

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.99713980841658,
    "longitude": -78.33071903597848,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()
print(f"{data}\n")

# Step 02: Transform
print(f"\nStep 02:\n")

daily = data["daily"]
records = [
    {
        "date":               daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum":  daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]
print(f"First record: {records[0]}\n")
print(f"Last record: {records[-1]}\n")
print(f"All records: {len(records)} records")

# I expected 365 records for each day and that's what printed out.

# Step 03: Load
print(f"\nStep 03:\n")

response = (
    supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
)
print(f"Upserted {len(response.data)} rows into weather_raw\n")
# Idemptency is very important for ensuring data reliability, safe retries, and efficiency.

# Step 04: Verify
print(f"\nStep 04:\n")

# row count
count_response = supabase.table("weather_raw").select("date", count="exact").execute()
print(f"Rows in weather_raw: {count_response.count}\n")

# First & last record
first = supabase.table("weather_raw").select("*").eq("date","2023-01-01").execute()
last = supabase.table("weather_raw").select("*").eq("date", "2023-12-31").execute()
specific_date = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()

print(f"First record: {first.data}\n")
print(f"Last record: {last.data}\n")
print(f"Record for July 4, 2023: {specific_date.data}\n")