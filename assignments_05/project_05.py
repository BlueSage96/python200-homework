from dotenv import load_dotenv
from openai import OpenAI

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