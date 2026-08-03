from dotenv import load_dotenv
from openai import OpenAI

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

#1. Each time the code is ran, each response is different from one another 
#   and different from the response of the run before
#2. I would choose the 1.5 temperature as it has produced the best names
#   has the most variation each run.

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

#System Question 02
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


#Prompt 01 - Zero Shot
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