from dotenv import load_dotenv
import os

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
