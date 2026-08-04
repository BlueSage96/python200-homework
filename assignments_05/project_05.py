from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# Task 1
messages_task1 = [
    {"role":"system","content":
      """
      You are a job application coach and you need to help job seekers 
      with polishing their resumes.
      
      Stay focused on the job application materials, even if the user 
      starts talking about personal stuff. 
      
      Always remind the user to review and edit your output before 
      submitting the resume anywhere.
      
      Even if you do not kow what the user's specific industry
      norms are, the user should use their own judgment.
      """
      }
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_task1,
    temperature=1.0
)

print(f"\nTask 01:\n")
print(f"Job builder:\n{response.choices[0].message.content}")

'''
I deliberately chose to instruct the chatbot to stay focused on its job 
even if the user starts talking about personal stuff. That way there 
is not a privacy or liability issue later on.
'''

# Task 02

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Respond ONLY with valid JSON, no other text.
    Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages,temperature=0)
    try: 
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        print("Error: response is not valid JSON")

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

new_bullets = rewrite_bullets(bullets)

print(f"\nTask 02:\n")
print(f"Bullet Point Rewriter:\n{new_bullets}")

# Addng try/catch didn't prevent errors! 
# I don't know how to fix the issue!!