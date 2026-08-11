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

# Step 5
print(f"\nStep 5:\n")
questions = [
    "Is Groundwork open on New Years Day?",
    "Do you offer any gluten-free food options?",
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
    1. I asked:  "Is Groundwork open on New Years Day?" and "Do you offer any gluten-free food options?".
       I made the query hard by changing a couple of questions for stuff not listed.
       
    2. For the first answer, the chatbot admitted it didn't know the answer and the chunk
       used (Our Story) was not relevant to the question.
       
       For the second answer, the chatbot fabricated the answer instead of saying it didn't
       know like it did for the first question. It also added the chunk for seasonal specials
       instead of the menu.
       
    3. The AI's tone remained remained matter-of-fact even when the answers were not available.
       AI-generated responses should always be vetted by humans to prevent harm from false information.
       
    4. I would train the AI to always admit that the information is not available instead of guessing.
       Also, I would change its tone to be uncertain to give it some "human" personality traits.
"""

# Step 6
"""
    1. My LlamaIndex code is 41 lines, and I believe it would have taken two times the code
       if I did semantic RAG manually. Using a framework saves a lot of time and headache
       especially for beginners.
       
    2. This system would work for any small business, like a small game studio, as the 
       chatbot is not intended to use for more complex questions.

    3. RAG cannot fully prevent the chatbot from providing misinformation for some queries.
"""

