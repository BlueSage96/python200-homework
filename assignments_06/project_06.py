from dotenv import load_dotenv
import os
import string
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Step 1
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
# Step 2
print(f"\nStep 02:\n")

docs_dir = Path("./resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

docs = SimpleDirectoryReader(docs_dir).load_data()
print(f"Documents loaded: {len(docs)}\n") 

dr = 0
for d in docs:
    dr += 1
    print(f"File {dr}: {d.metadata["file_name"]}")