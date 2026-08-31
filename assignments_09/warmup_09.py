import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import date

# --- Supabase Connection ---
# Q1

# 1. supabase-py needs the Supabase project URL and API key to connect to the project.

# 2. The Supabase project URL can be found in Project Settings under the API section.
#    The API key can also be found in the API section of the Supabase dashboard.

#    These values should not be hardcoded in the source code or committed to a public
#    GitHub repository because exposing credentials can allow unauthorized access to
#    the Supabase project.

# Q2

def get_client():
    load_dotenv()  # reads .env and sets environment variables
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")   
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variable")
        
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

# Q3

# Row Level Security (RLS) is a Supabase/PostgreSQL security feature that controls
# which rows users can access or modify based on policies.

# RLS is normally important in production applications because it can restrict
# users to only the data they are authorized to access. It is disabled for this
# course to simplify development and allow the Python program to insert and
# access the weather data without creating additional RLS policies.

# --- CRUD ---
# Q1

def insert_test_record(supabase):
    record = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 16.3,
        "temperature_2m_min": 2.1,
        "precipitation_sum": 0.7,
        "wind_speed_10m_max": 15.5,
    }
    
    response = supabase.table("weather_raw").insert(record).execute()
    print(response.data)

# Running this function twice with insert() would cause a duplicate key error
# because date is the primary key. Using upsert() instead would make the operation
# safe to repeat by updating the existing row when the date already exists.

supabase = get_client()
insert_test_record(supabase)
# Q2

def get_records_by_date_range(supabase, start, end):
    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    rows = response.data  # list of dicts
    return rows

records = get_records_by_date_range(supabase, "2020-01-01", "2026-08-30")

print(f"\n CRUD 02:\n")
print(records)

# Q3

# Using a plain "insert" can result in an error if the record already exists. 
# Upsert inserts a new row if the key is new or updates the existing row if 
# it doesn't already exist.

# Example: A user creates a sudoku game picking the difficulty. When they are finished, 
# the record updates to include the number of hints used and the number of errors for that 
# game. Insert would work well to insert this data into the database; however, if this is a 
# game the user lost (too many mistakes) and they retry again, upsert needs to be used to 
# update the existing record. 

def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    )
    print(f"\nRows affected: {len(response.data)}\n")
    
safe_upsert(supabase, records)

# --- Idempotency ---
# Q1

# Idempotency is important in data pipelines because it allows a failed pipeline
# to be safely restarted without creating duplicate records or corrupting data.
# If a pipeline crashes halfway through loading data and is restarted, an idempotent
# operation can update existing records instead of inserting duplicates.

# Example: A Sudoku game update fails after the game reaches the maximum number
# of mistakes. When the update is retried, idempotency ensures the existing game
# record is updated instead of creating a duplicate game record.