from dotenv import load_dotenv
from openai import OpenAI
import json

#API 01
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [{"role":"user","content":
        "What is one thing that makes Python a good language for beginners?"}]
)

print(f"API 01:\n")
print(f"Response: \n{response.choices[0].message.content}")
print(f"\nModel: {response.model}")
print(f"\nTotal tokens:{response.usage.total_tokens}")

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
1.  When I tested different temperature settings, the responses became 
    more creative as the temperature increased. Temperature 0 produced the 
    most consistent names, while 1.5 generated the most varied and unexpected 
    suggestions.
    
2.  I would choose a temperature of 0 when I need consistent and reproducible 
    output because it produced the most predictable results across multiple runs. 
    Higher temperatures introduced more variation, which is useful for brainstorming 
    but not for repeatable responses.
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
    With max_tokens set to 15, the model's explanation was cut off before it could fully
    answer the question. This demonstrated how limiting the token count truncates longer
    responses.

    Using max_tokens is useful because it helps control API costs, keeps responses concise,
    and prevents the model from generating more text than an application needs.
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

'''
    The model remembered Jordan's name because the previous conversation was included in the 
    messages list with each API call. The Chat Completions API is stateless, so it only has 
    access to information that is explicitly provided in the conversation history.
'''


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
print("Zero Shot")
# No examples provided - model uses pre-trained knowledge
z = 0
for r in reviews:
    z += 1
    prompt = f"""Classify the sentiment of this review as positive, negative, or mixed.
                Review: "{r}"
                Sentiment:
            """
    response = get_completion(prompt, temperature=0)
    print(f"Review {z}:\n{response}")
    
# Prompt 02 - One Shot
print(f"\nPrompt 02:\n")
print("One-Shot")
o = 0
for r in reviews:
    o += 1
    prompt = f"""
        Classify the sentiment of each review as positive, negative, or mixed.

        Example:
        Review: "The new laptop turns on but has a blank screen."
        Sentiment: Mixed

        Now classify this review:

        Review: "{r}"
        Sentiment:
        """
    response = get_completion(prompt, temperature=0)
    print(f"Review {o}:\n{response}")
'''
    The zero-shot prompt correctly classified all three reviews using the model's existing
    knowledge, but the wording of the responses varied slightly. The one-shot example produced
    more consistent formatting because the model had a reference to imitate. The few-shot
    prompt produced the most consistent labels because multiple examples reinforced the
    expected pattern.
'''

# Prompt 03 - Few-Shot
print(f"\nPrompt 03:\n")
print("Few-Shot")
fs = 0
for r in reviews:
    fs += 1
    prompt = f"""
        Classify the sentiment of each review as positive, negative, or mixed.

        Example 1:
        Review: "The staff was friendly and solved my issue quickly."
        Sentiment: positive

        Example 2:
        Review: "The software crashes every time I try to save my work."
        Sentiment: negative

        Example 3:
        Review: "The price was excellent, but the instructions were difficult to follow."
        Sentiment: mixed

        Now classify this review:

        Review: "{r}"
        Sentiment:
    """
    response = get_completion(prompt, temperature=0)
    print(f"Review {fs}:\n{response}")
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
    Provide a brief explanation, then give the final answer on its own line labelled: Final answer: 
    
    Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
    takes a new job that pays $7,500 more per year than her post-raise salary.
    What is her final annual salary?
"""
response = get_completion(prompt, temperature=0)

print("Raw response:")
print(response)

try:
    result = json.loads(response)

    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("Error: Unable to parse the model's response as JSON.")
    print("Raw response:")
    print(response)

'''
The brief explanation helps show how the model reached the answer and makes the calculation
easier to verify. Asking for the final answer on its own labeled line also makes the result
easy to identify.
'''

# Prompt 05 - Structured Output

print(f"\nPrompt 05:\n")

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
prompt = f"""
        Classify the sentiment of the review.

        Return ONLY valid JSON.

        Do not include markdown.
        Do not include code fences.
        Do not include explanations.
        Do not include any text before or after the JSON.

        Return exactly this structure:

        {{
            "sentiment": "positive | negative | mixed",
            "confidence": 0.0,
            "reason": "one short sentence"
        }}

        Review:
        {review}
    """
response = get_completion(prompt, temperature=0)

print("Raw response:")
print(response)

try:
    result = json.loads(response)

    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("JSON parsing failed.")
    print("Raw response:")
    print(response)
    

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

'''
The first prompt was correctly converted into a numbered list because it contained
step-by-step instructions. The second prompt was recognized as prose, and the
model responded with exactly "No steps provided." as instructed.
'''

# Ollama 01
prompt = f"""
Review: Explain what a large language model is in two sentences.
"""
response = get_completion(prompt, temperature=0)
print(f"\nOpenAI response:\n {response}")

# Ollama terminal output
'''
Ollama terminal output:

A large language model is a complex system that processes and generates human-like text, enabling it to
understand and respond to language in real time. It is trained on vast amounts of text, allowing it to learn
patterns and general knowledge, making it capable of tasks like writing, speaking, or even answering questions.
'''

'''
1. Both models correctly explained what a large language model is in two sentences.
   The OpenAI response included a little more explanation, while the Ollama response
   was shorter and more concise.

2. One advantage of running a local model is that prompts and responses stay on your
   own computer, providing greater privacy and reducing reliance on an internet connection
   or third-party service.

3. One disadvantage of local models is that they require more powerful hardware,
   storage, and memory than many cloud-based AI services, which can make them more
   expensive to run.
'''