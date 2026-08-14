from dotenv import load_dotenv
import os
import string
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
    
# Concepts 01

# Scenario A:
# RAG makes the most sense because there are hundreds of PDFs that are frequently
# updated. The system can retrieve relevant information from the current internal
# documents when answering a user's question without retraining the model.

# Scenario B:
# Fine-tuning makes the most sense because the startup wants the AI to consistently
# adopt a particular style and personality. Fine-tuning can teach the model desired
# patterns and behaviors that would be difficult to express entirely through prompts.

# Scenario C:
# Prompt engineering makes the most sense because the data analyst only needs to
# work with a small report. A carefully written prompt can provide the necessary
# instructions and context without requiring RAG or fine-tuning.


# Concepts 02

# AI hallucination can be harmful if a person asks an AI for medical advice.
# If the AI confidently gives incorrect medical advice, the person could follow
# the advice and become seriously injured or even die. A confident tone can
# increase trust because users may interpret certainty as a sign that the AI
# knows what it is talking about. This can make users less likely to question
# or verify the response, increasing the chance that they will act on false
# information.


# Concepts 03

# Original:
#     steps = [
#         "Generate a response from the LLM",
#         "Extract text from source documents",
#         "Receive the user's query",
#         "Retrieve the most relevant chunks",
#         "Convert text chunks into embeddings",
#         "Inject retrieved chunks into the prompt",
#         "Split text into chunks",
#         "Embed the user's query",
#     ]

#  Fixed:

#    steps = [

#     "Receive the user's query — The system receives the question the user wants answered.",
#     "Embed the user's query — The query is converted into an embedding so it can be compared with document embeddings.",
#     "Extract text from source documents — Text is extracted from the documents that will provide information for the response.",
#     "Split text into chunks — The extracted text is divided into smaller chunks that can be searched efficiently.",
#     "Retrieve the most relevant chunks — The system finds the document chunks that are most similar to the user's query.",
#     "Convert text chunks into embeddings — The document chunks are converted into embeddings so their meaning can be compared with the query.",
#     "Inject retrieved chunks into the prompt — The relevant chunks are added to the prompt as context for the LLM.",
#     "Generate a response from the LLM — The LLM uses the user's question and retrieved context to generate the final response."

#    ]


#  Conclusion:
 
#  The RAG steps breaks large files of data into digestable 
#  chunks that only contains the most important information so 
#  that the model can give the most accurate response to the 
#  user's query.


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
# Keywords 01

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

keywords = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Keywords 01:\n")
print(keywords)

# loyalty.txt was selected because keyword retrieval found one exact keyword
# match in loyalty.txt ("your"). hours.txt also had one matching keyword
# ("weekends"), so the keyword-based method could not distinguish which
# document was more relevant.

# Keywords 02

query = "Do you have anything without caffeine?"
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
keywords = simple_keyword_retrieval(query, documents, verbose=True)
print(f"\nKeywords 02:\n")
print(keywords)


# Keyword RAG struggled because there were no overlapping keywords.
# Semantic retrieval would work better because embeddings can identify similar
# meanings even when the query and document use different words.


# Prediction: My first pick would be loyalty.txt as its 
# content is more aligned with the query below. 
# But I have a feeling it would be hiring or hours instead.

query = "How do I sign up for rewards?"
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
keywords = simple_keyword_retrieval(query, documents, verbose=True)
print(f"\nKeywords 03:\n")
print(keywords)

# My prediction was wrong because there were no overlapping keywords, so no document could be selected.

# Semantic 01


# 1. A vector embedding is a numerical representation of the meaning of text,
# created by an embedding model so that text can be compared mathematically.

# 2. A cosine similarity score of 0.85 indicates greater semantic similarity
# than a score of 0.30. Therefore, the chunk with a score of 0.85 is more
# relevant to the query.

# 3. Semantic retrieval can find relevant chunks even when the query and the
# retrieved text do not contain the same exact words because their meanings
# can still be similar.


# Semantic 02

# | Feature                    | Keyword RAG                              | Semantic RAG                                      |
# |----------------------------|------------------------------------------|---------------------------------------------------|
# | What is compared?         | Exact words in the query and documents  | Vector embeddings representing meaning              |
# | What is retrieved?        | Matching document/content               | Most semantically similar text chunks               |
# | Can it handle synonyms?   | No                                       | Yes                                                |
# | Storage format             | Dictionary of documents and text        | Vector index containing document chunk embeddings  |
# | Relevance score            | Number of matching keywords              | Cosine similarity between embeddings              |


# LlamaIndex 01

# Load documents directly from PDFs in the folder
brightleaf_dir = Path(__file__).resolve().parent / "brightleaf_pdfs"
docs = SimpleDirectoryReader(brightleaf_dir).load_data()

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

query_engine_k3 = index.as_query_engine(similarity_top_k=3)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

print(f"\nLlamaIndex 01:\n")

for q in questions:
    # Q & A
    print(f"\nQ: {q}")
    response = query_engine_k3.query(q)
    print(f"\nA: {response}")
    
    for node_with_score in response.source_nodes:
        # Retrieve 3 retreived source nodes
        print(f"\nNODE ID: {node_with_score.node_id}")
        print(f"\nSimilarity Score: {node_with_score.score:.4f}")
        print(f"\nText Snippet: {node_with_score.node.get_content()[:150]}")
        print("-" * 30)
        

# The answers to each question are specific and relevant to the questions.
# The tone of the answers are professional. There no unexpected outputs.


# LlamaIndex 02
print(f"\n========== LlamaIndex 02: similarity_top_k=1 ==========\n")

query_engine_k1 = index.as_query_engine(similarity_top_k=1)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    # Q & A
    print(f"\nQ: {q}")
    response = query_engine_k1.query(q)
    print(f"\nA: {response}")
    
    for node_with_score in response.source_nodes:
        # Retrieve 3 retreived source nodes
        print(f"\nNODE ID: {node_with_score.node_id}")
        print(f"\nSimilarity Score: {node_with_score.score:.4f}")
        print(f"\nText Snippet: {node_with_score.node.get_content()[:150]}")
        print("-" * 30)
        
print(f"\n========== LlamaIndex 02: similarity_top_k=5 ==========\n")

query_engine_k5 = index.as_query_engine(similarity_top_k=5)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    # Q & A
    print(f"\nQ: {q}")
    response = query_engine_k5.query(q)
    print(f"\nA: {response}")
    
    for node_with_score in response.source_nodes:
        # Retrieve 3 retreived source nodes
        print(f"\nNODE ID: {node_with_score.node_id}")
        print(f"\nSimilarity Score: {node_with_score.score:.4f}")
        print(f"\nText Snippet: {node_with_score.node.get_content()[:150]}")
        print("-" * 30)
        
# k=1 produced the most relevant chunk, while k=5 produced 5 chunks,
# some of them less relevant. Semantic RAG shows the most relevant
# chunk followed by the less relevant ones. The similarity scores for
# k=1, k=3, and k=5 are consistent with one another. More retrieved
# context is not always better.

# LlamaIndex 03
print(f"\nLlamaIndex 03:\n")
query_engine = index.as_query_engine(similarity_top_k=3)
questions = [
    "What is the product feedback from BrightLeaf's customers?",
    "What type of benefits are missing from BrightLeaf's employee benefits?",
    "Does BrightLeaf's security policy involve any mention of AI enhanced security measures?"
]

for q in questions:
    # Q & A
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print(f"\nA: {response}")
    
    for node_with_score in response.source_nodes:
        # Retrieve 3 retreived source nodes
        print(f"\nNODE ID: {node_with_score.node_id}")
        print(f"\nSimilarity Score: {node_with_score.score:.4f}")
        print(f"\nText Snippet: {node_with_score.node.get_content()[:150]}")
        print("-" * 30)
        

# Reflection:

# I expected these questions to be difficult because the BrightLeaf PDFs
# do not contain direct information about customer product feedback, missing
# employee benefits, or AI-enhanced security measures.

# The first two questions produced responses that were not directly supported
# by the retrieved documents. The third question retrieved a security-related
# chunk that was relevant to the topic, but it did not establish that
# BrightLeaf uses AI-enhanced security measures.

# This shows that retrieving text that is related to a question does not
# guarantee that the text actually answers the question. I would improve
# the system by requiring stronger evidence from the retrieved chunks and
# having it say that the information is unavailable when the documents do
# not support an answer.

# LlamaIndex 04
print(f"\nLlamaIndex 04:\n")

# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

print(f"\nLlamaIndex 04 employee benefits:\n")
# Get response to query
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

print(f"\nLlamaIndex 04 produce feedback:\n")
# Get response to query
q = "Do you think Grand Theft Auto VI be delayed again?"
response = query_engine.query(q)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Reflection:

# For the employee-benefits query, the faithfulness score was 1 and the
# relevancy score was 1. For the unrelated GTA VI query, the faithfulness
# score was 0 and the relevancy score was 0.

# The lower scores on the unrelated query show that the retrieved BrightLeaf
# context did not provide useful evidence for answering the question.
# Faithfulness measures whether the response is supported by the retrieved
# context, while relevancy measures whether the response addresses the query.

# This demonstrates why evaluator scores can help identify when a RAG system
# is producing responses that are poorly supported or unrelated to its
# source documents.