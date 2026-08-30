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

get_client()