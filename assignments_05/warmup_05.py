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