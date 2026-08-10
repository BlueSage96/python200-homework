from dotenv import load_dotenv
import os
import string
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
docs_dir = Path("./resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
