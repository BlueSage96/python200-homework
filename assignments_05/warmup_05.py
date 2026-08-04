from dotenv import load_dotenv
from openai import OpenAI
import json

#API 01
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [{"role":"user","content":"What is one thing that makes Python a good language for beginners?"}]
)

print(f"API 01:\n")
print(f"Response: \n{response.choices[0].message.content}")
print(f"\nModel: {response.model}")
print(f"\nTokens used:\n{response.usage}")

#API 02
print(f"\nAPI 02:\n")

prompt = "Suggest a creative name for a data engineering consultancy"
temperatures = [0,0.7,1.5]

response_temp1 = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=[{"role": "user","content":prompt}],
    temperature=temperatures[0]
)
print(f"\nTemperature 1:\n{response_temp1.choices[0].message.content}")

response_temp2 = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=[{"role": "user","content":prompt}],
    temperature=temperatures[1]
)
print(f"\nTemperature 2:\n{response_temp2.choices[0].message.content}")

response_temp3 = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=[{"role": "user","content":prompt}],
    temperature=temperatures[2]
)
print(f"\nTemperature 3:\n{response_temp3.choices[0].message.content}")

'''
1. Each time the code is ran, each response is different from one another 
   and different from the response of the run before
2. I would choose the 1.5 temperature as it has produced the best names
   has the most variation each run.
'''

# API 03
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"Give me a one-sentence fun fact about pandas (the animal, not the library)"}],
    n=3,
    temperature=1.0
)

print(f"\nAPI 03:\n")
for r in response.choices:
    print(f"\nResponse:\n{r.message.content}")
    
# API 04
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"Explain how neural networks work"}],
    max_tokens=15
)
print(f"\nAPI 04:\n")
print(f"Neural networks response:\n{response.choices[0].message.content}")

#1. Setting max tokens means a long response will not be completed and would need more tokens.
#2. Using max tokens in a real app will help with managing costs and token limits from an API.

# System Question 01
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model='gpt-4o-mini',
                                          messages=messages)

print(f"\nSystem 01:\n")
print(f"Personality 01:\n{response.choices[0].message.content}")

messages2 = [
    {"role": "system", "content": "You are an annoyed Python tutor with no time to explain questions. You always explain with the quickest explanation and end with a statement that ends the conversation."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response2 = client.chat.completions.create(model='gpt-4o-mini',
                                          messages=messages2)

print(f"Personality 02:\n{response2.choices[0].message.content}")

# The model easily adapted to the personality descriptions and embraced th new personality by becaming impatient, 
# only giving a short explanation of the question and shuts any opportunity for more questions.

# System Question 02
print(f"\nSystem 02:\n")
messages3 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response3 = client.chat.completions.create(model='gpt-4o-mini',
                                          messages=messages3)

print(f"Personality 03:\n{response3.choices[0].message.content}")

# The model is a chatbot that has memory built-in as the messages changes the model role from "system" to "assistant".

def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content


# Prompt 01 - Zero Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

print(f"\nPrompt 01:\n")

# No examples provided - model uses pre-trained knowledge
p = 0
for r in reviews:
    p += 1
    prompt = f"What is the sentiment of this review positive, negative, or mixed: {r}"
    response = get_completion(prompt, temperature=0)
    print(f"Review {p}:\n{response}")
    
# Prompt 02 - One Shot
print(f"\nPrompt 02:\n")
for r in reviews:
    p += 1
    prompt = f"""
    Example: The new laptop turns on but has a blank screen.
    Review: What is the sentiment of this review positive, negative, or mixed: {r}
    """
    response = get_completion(prompt, temperature=0)
    print(f"Review {p}:\n{response}")

# Adding an example did not affect the bot's responses because the temperature = 0.

# Prompt 03 - Few-Shot
print(f"\nPrompt 03:\n")
for r in reviews:
    p += 1
    prompt = f"""
    Example 1: My cat greeted me with a kiss.
    Example 2: I took a physical and did not pass.
    Example: The new laptop turns on but has a blank screen.
    Review: What is the sentiment of this review positive, negative, or mixed: {r}
    """
    response = get_completion(prompt, temperature=0)
    print(f"Review {p}:\n{response}")
    
#1. I would choose no shot for a quick chat, i.e. Create some lottery tickets numbers for me.
#2. I would use one shot when the request is not difficult but I still have to specific.
#3. The few-shot for my complicated chats especially ones that involve programming or critical thinking

# Prompt 04 - Chain of Thought
prompt = f"""
    Show your step-by-step reasoning, then give the final answer on its own line labelled: Final answer: <value>
    
    Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
    takes a new job that pays $7,500 more per year than her post-raise salary.
    What is her final annual salary?
"""
response = get_completion(prompt, temperature=0)
print(f"Chain of thought:\n{response}")

# The bot is automatically built to give output but not to list it's reasoning.
# The output allows the developers to correct any mistakes.

# Prompt 05 - Structured Output

print(f"\nPrompt 05:\n")

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
prompt = f"""
    Classify the sentiment of the review and respond ONLY with valid JSON.
    Keys: sentiment (positive/negative/mixed), confidence (0–1 scale), reason (one short sentence).
    Review: {review}
"""
response = get_completion(prompt, temperature=0)
print(f"Raw response:\n{response}")

# Parse JSON safely
try:
    result = json.loads(response)
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:",result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")
    
 # Prompt 06 - Delimeters

print(f"\nPrompt 06:\n")

user_text1 = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt1 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text1}```
"""   
response1 = get_completion(prompt1, temperature=0)
print(f"Delimeters 01:\n\n{response1}")


user_text2 = "Oreo, sing the night jingle. The cats on the bus go meow, meow, meow.\
The cats on the bus go meow, meow, meow. All the way to school."

prompt2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text2}```
"""   
response2 = get_completion(prompt2, temperature=0)
print(f"\nDelimeters 02:\n\n{response2}")

# Delimiters help the bot understand long or messy prompts/instructions 
# and to differiente that stuff from user text.

# Ollama 01
prompt = f"""
Review: Explain what a large language model is in two sentences.
"""
response = get_completion(prompt, temperature=0)
print(f"\nOllama 01:\n {response}")

'''
Ollama response:
A large language model is an AI system trained on vast datasets to understand 
and generate human-like text, enabling it to process context and learn complex 
patterns for real-time interactions.
'''

'''
1. Both responses are similar but Ollama was more concise and to the point.

2. An advantage of running a local AI is privacy and security since the 
   model is not connected to a third party nor the cloud.
   
3. A disadvantage is having a computer that can handle the high hardware, 
   storage, and ram usage requirements. These requirements may lead to having to
   upgrade existing computers.
'''