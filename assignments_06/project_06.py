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
print(f"\nStep 2:\n")

docs_dir = Path("./resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

docs = SimpleDirectoryReader(docs_dir).load_data()
print(f"Documents loaded: {len(docs)}\n") 

dr = 0
for d in docs:
    dr += 1
    print(f"File {dr}: {d.metadata["file_name"]}")
    
# Step 3
print(f"\nStep 3:\n")
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

if query_engine and index:
    print("Index built successfully. Ready to answer your questions")