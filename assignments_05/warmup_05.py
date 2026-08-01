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