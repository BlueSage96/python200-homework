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
    
# Step 4
print(f"\nStep 4:\n")
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

query_engine = index.as_query_engine(similarity_top_k=1)

for q in questions:
    print(f"\nQuestion: {q}")
    response = query_engine.query(q)
    print(f"\nAnswer: {response}\n")
    
    for node_with_score in response.source_nodes:
        # Retrieve the top retreived source node
        print(f"\nNODE ID: {node_with_score.node_id}")
        print(f"\nSimilarity Score: {node_with_score.score:.4f}")
        print(f"\nText Snippet: {node_with_score.node.get_content()[:200]}")
        print("-" * 30)
        
"""
    The chatbot was very confident and accurate, giving concise answers.
    There were two issues with a couple of the responses:
    
    - Answer two should have come from the menu.txt as there are non-seasonal
      dairy-free drinks.
    - Answer five is vague about the catering and wholesale orders and 
      could have briefly explained them better.
      
    There were no "surprising" answers, just two bad ones.
"""
