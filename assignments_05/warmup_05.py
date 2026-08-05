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
1. When I tested different temperature settings, the responses became more 
   creative as the temperature increased. Temperature 0 produced the most 
   consistent names, while 1.5 generated the most varied and unexpected 
   suggestions.
2. I would choose a temperature of 1.5 for brainstorming because it consistently 
   gave me the most creative and unique consultancy names.
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

'''
1. With max_tokens set to 15, the explanation was cut off before it could fully 
   explain how neural networks work. This showed how limiting the token count can 
   shorten or truncate a response.
2. Using max_tokens in a real application helps control response length, reduce
   API costs, and keep responses concise when a long explanation isn't necessary.
'''

# System Question 01
messages = [
    {"role": "system", "content": 
        """
        You are a patient, encouraging Python tutor. 
        You always explain things simply and end with a word of encouragement."""},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model='gpt-4o-mini',
                                          messages=messages)

print(f"\nSystem 01:\n")
print(f"Personality 01:\n{response.choices[0].message.content}")

messages2 = [
    {"role": "system", "content": """
     You are an annoyed Python tutor with no time to explain questions. 
     You always explain with the quickest explanation and end with a statement that 
     ends the conversation."""},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response2 = client.chat.completions.create(model='gpt-4o-mini',
                                          messages=messages2)

print(f"Personality 02:\n{response2.choices[0].message.content}")

'''
    The two system prompts produced noticeably different responses even though the user 
    asked the exact same question. The first tutor gave a detailed, encouraging explanation 
    and ended with positive encouragement, while the second tutor gave a much shorter, impatient 
    response and ended the conversation instead of inviting further questions. This showed that 
    the system prompt strongly influences the model’s tone, teaching style, and overall behavior.
'''


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
z = 0
for r in reviews:
    z += 1
    prompt = f"What is the sentiment of this review positive, negative, or mixed: {r}"
    response = get_completion(prompt, temperature=0)
    print(f"Zero shot 0{z}:\n{response}")
    
# Prompt 02 - One Shot
print(f"\nPrompt 02:\n")
o = 0
for r in reviews:
    o += 1
    prompt = f"""
    Example: The new laptop turns on but has a blank screen.
    Review: What is the sentiment of this review positive, negative, or mixed: {r}
    """
    response = get_completion(prompt, temperature=0)
    print(f"One-Shot 0{o}:\n{response}")

'''
The one-shot prompt gave the model a reference for the expected format. Although the results 
were similar to the zero-shot responses for this task, providing an example can make the 
output more consistent for more complex prompts.
'''

fs = 0
# Prompt 03 - Few-Shot
print(f"\nPrompt 03:\n")
for r in reviews:
    fs += 1
    prompt = f"""
    Example 1: My cat greeted me with a kiss.
    Example 2: I took a physical and did not pass.
    Example: The new laptop turns on but has a blank screen.
    Review: What is the sentiment of this review positive, negative, or mixed: {r}
    """
    response = get_completion(prompt, temperature=0)
    print(f"Few-Shot 0{fs}:\n{response}")
    
'''
1. The zero-shot prompt worked well for this simple sentiment classification task 
   because the model already understood the request without needing examples. It produced 
   short and consistent labels for each review.

2. The one-shot prompt gave the model a reference for the expected format. Although the 
   results were similar to the zero-shot responses for this task, providing an example can 
   make the output more consistent for more complex prompts.

3. The few-shot prompt provided multiple examples, making the expected pattern even clearer. 
   For this simple sentiment task, the answers were similar to the other approaches, but few-shot 
   prompting is more useful when a task requires a specific style, format, or reasoning pattern.

4. I would use zero-shot for simple, common tasks that the model already understands, such as 
   sentiment classification or basic questions. I would use one-shot when I want to demonstrate 
   the expected format with a single example. I would use few-shot for more complex tasks where 
   several examples help the model produce more consistent and accurate responses.
'''

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

# Parse JSON safely
try:
    result = json.loads(response)   
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:",result["reason"])
except json.JSONDecodeError:
    print(f"Error: Unable to parse the model's response as JSON. Raw response:{response}")
    
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


user_text2 = """Oreo is a domestic shorthair tuxedo cat and this is his bedtime lullaby. 
The cats on the bus go meow, meow, meow.\
The cats on the bus go meow, meow, meow. All the way to school."""

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
print(f"\nOpenAI response:\n {response}")

# Ollama
'''
A large language model is a complex system that processes and generates human-like text, enabling it to 
understand and respond to language in real time. It is trained on vast amounts of text, allowing it to learn 
patterns and general knowledge, making it capable of tasks like writing, speaking, or even answering questions.
'''

'''
1. Both responses are similar but Ollama was more concise and to the point.

2. An advantage of running a local AI is privacy and security since the 
   model is not connected to a third party nor the cloud.
   
3. A disadvantage is having a computer that can handle the high hardware, 
   storage, and ram usage requirements. These requirements may lead to having to
   upgrade existing computers.
'''