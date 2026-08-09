from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
    
# Concepts 01

"""
    Scenario A:
    RAG makes the most sense since there are hundreds of PDFs that are updated.
    The team can train the AI with their internal library and retrieve outside 
    documents when necessary.

    Scenario B:
    Since the startup has produced a lot of information from internal files,
    fine tuning is the way because they want the AI to adopt a certain 
    style/personality. Another reason would be to accomodate internalize patterns 
    that are too complex to express in a prompt. 
    
    Scenario C:
    For a small report, prompt engineering is the best option as the data analyst
    can give a detailed prompt and let the model answer.
"""

# Concepts 02

"""
    AI hallucination can be harmful if the human asks for medical advice.
    If the AI confidently gives the user the wrong medical advice, 
    the person would possibly get hurt or killed. Even worse, the 
    confident bot can insist on the user to take the medical advice
    and it could reject any corrections. The bot's tone would be
    arrogrant because it thinks it is a medical professional, not 
    a sentient 
"""

# Concepts 03

"""
  Original:
    steps = [
        "Generate a response from the LLM",
        "Extract text from source documents",
        "Receive the user's query",
        "Retrieve the most relevant chunks",
        "Convert text chunks into embeddings",
        "Inject retrieved chunks into the prompt",
        "Split text into chunks",
        "Embed the user's query",
    ]

 Fixed:
 
    steps = [
        "Receive the user's query",
        "Embed the user's query",
        "Extract text from source documents",
        "Split text into chunks",
        "Retrieve the most relevant chunks",
        "Convert text chunks into embeddings",
        "Inject retrieved chunks into the prompt",
         "Generate a response from the LLM",
    ]

 Conclusion:
 
 The RAG steps breaks large files of data into digestable 
 chunks that only contains the most important information so 
 that the model can give the most accurate response to the 
 user's query.
"""



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
# loyalty.txt because the documents are being ranked based on the number of exact keyword matches with the query. 
# In this case, the keyword is "your".

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

# There were no overlapping keywords found so keyword RAG was wrong for this query. Fine-tuning would be more appropriate for this example.

# Keywords 03

"""
    Prediction: My first pick would be loyalty.txt as its 
    content is more aligned with the query below. 
    But I have a feeling it would be hiring or hours instead.
"""
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

"""
    1. A vector embedding is the process where chunks of data is converted to numbers by the embedding model.
    2. The score of 0.85 is more relevant as it is closest to being similar (almost 1). Both scores (0.85 & 0.30)
       have some mild similarity to one another for the model to still associate them together.
    3. Even if the relevant chunks do not share text with one another, they can have a similar meaning.
"""