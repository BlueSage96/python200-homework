import os
from dotenv import load_dotenv
from supabase import create_client

# Connection 01

# 1. supabase-py needs the supabase url and key to connect to the project.

# 2. Both the supabase url can be accessed by hovering over the settings (gear icon) 
#    and selecting "Project Settings". The url is the "Project ID".

#    The supabase key can be found by clicking on "API keys" on the project's 
#    main dashboard under the "Get Connected".

#   We should never commit a key to a public GitHub repository because it can be scraped 
#   within minutes by automated bots. 

# Connection 02

def get_client():
    if load_dotenv():  # reads .env and sets environment variables
        print('Successfully loaded environment varables from .env')
    else:
        print('Warning: could not load environment variables from .env')
        
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    return supabase

# Connection 03

# Row Level Security (RLS) dictates what table rows a use can access. 
# RLS is an important production feature, but it adds complexity during development. 

# A real-world use case would be making an important distinction between the admin and 
# regular users in an app. The admin can access all of the data in the app while 
# regular users can only see the data they created.

# CRUD 01

def insert_test_record(supabase):
    record = {
        "date":"2026-08-30",
        "temperature_2m_max": 16.3,
        "temperature_2m_min": 2.1,
        "precipitation_sum":  0.7,
        "wind_speed_10m_max": 15.5,
    }
    
    response = supabase.table("weather_raw").insert(record).execute()
    print(response.data)

supabase = get_client()
insert_test_record(supabase)

# Running the function twice creates a 'duplicate key' error.
# Would use "upsert" in the function instead of "insert" to 
# make repeated calls safe.

# CRUD 02

def get_records_by_date_range(supabase, start, end):
    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    rows = response.data  # list of dicts
    return rows

records = get_records_by_date_range(supabase, "2020-01-01", "2026-08-30")

print(f"\n CRUD 02:\n")
print(records)

# CRUD 03

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
    print(f"\nRows effected: {response}\n")
    
safe_upsert(supabase, records)

# Idempotency 01

# Idempotency preserves pipeline data by not duplicating it. If data fails to load while 
# running the pipeline, idempotency will prevent the original data from being lost.
# Example: Retrying a failed sudoku game (hits max mistakes) with idempotency updating 
# the existing record instead of duplicating it.