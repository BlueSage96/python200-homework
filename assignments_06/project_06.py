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

docs_dir = "./resources/groundwork_docs"
assert os.path.exists(docs_dir), "Groundwork documents directory does not exist."

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

query_engine = index.as_query_engine(similarity_top_k=3)

for q in questions:
    print(f"\nQuestion: {q}")
    response = query_engine.query(q)
    print(f"\nAnswer: {response}\n")

    top_node = response.source_nodes[0]
    print(f"Top Retrieved Document: {top_node.node.metadata.get('file_name')}")
    print(f"Similarity Score: {top_node.score:.4f}")
    print(f"Text Snippet: {top_node.node.get_content()[:200]}")
    print("-" * 30)
        

# The chatbot was very confident and accurate, giving concise answers.
# There were two issues with a couple of the responses:

# Answer two should have come from the menu.txt as there are non-seasonal
# dairy-free drinks.
# Answer five is vague about the catering and wholesale orders and 
# could have briefly explained them better.
    
# There were no "surprising" answers, just two bad ones.


# Step 5
print(f"\nStep 5:\n")
questions = [
    "Is Groundwork open on New Years Day?"
]

query_engine = index.as_query_engine(similarity_top_k=3)

for q in questions:
    print(f"\nQuestion: {q}")
    response = query_engine.query(q)
    print(f"\nAnswer: {response}\n")
    
    for node_with_score in response.source_nodes:
        # Print all three retrieved source nodes required for the failure analysis.
        print(f"\nDocument: {node_with_score.node.metadata.get('file_name')}")
        print(f"\nSimilarity Score: {node_with_score.score:.4f}")
        print(f"\nText Snippet: {node_with_score.node.get_content()[:200]}")
        print("-" * 30)
        

# Step 5 Reflection:

# I asked, "Is Groundwork open on New Year's Day?" because the Groundwork
# documents do not provide information about holiday hours, so I expected
# the system to struggle with the question.

# The retrieved documents did not contain the answer. The model acknowledged 
# the missing information saying that there is no information on New Year's
# Day.

#The model remained confident when explaining that the information was
# unavailable. This shows that an AI can sound confident even when 
# the retrieved information does not support its answer, so users should 
# verify important information rather than relying on confidence alone.

# I would improve the system by adding a similarity threshold or a
# "not enough information" safeguard so that the system can decline to
# answer when the retrieved documents do not provide sufficient evidence.


# Step 6

# 1. My LlamaIndex code is 41 lines, and I believe it would have taken two times the code
#    if I did semantic RAG manually. Using a framework saves a lot of time and headache
#    especially for beginners.
    
# 2. This system would work for any small business, like a small game studio, as the 
#    chatbot is not intended to use for more complex questions.

# 3. RAG cannot fully prevent the chatbot from providing misinformation for some queries.