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