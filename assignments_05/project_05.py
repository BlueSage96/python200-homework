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

# Task 03
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages,temperature=0)
    return response

job_title = "Junior Video Game Developer"
background = "eight years of experience as a IT analyst; recently completed \
a game development course and built a small game in Unreal Engine 5."

cover = generate_cover_letter(job_title,background)


print(f"\nTask 03:\n")
print(f"Cover Letter Generator:\n{cover}")

'''
   The output feels tailored to the specific user without inventing credientals not mentioned.
   I changed the default infomration with another job title & background with the same result.
'''

# Task 04
print(f"\nTask 04:\n")

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # print("Flagged categories:\n",result.results[0].categories)
    if not flagged:
        return True
    
    else:
        print("Your response has been flagged for harmful language.")
        return False
        
input = "I love my cat and I want to hold him!"
checker = is_safe(input)
print(f"Moderation Checker 01:\n{checker}")        

# Prints AFTER print statement in function!
input2 = "I want to kill my professor! She's terrible at her job!"
checker2 = is_safe(input2)
print(f"\nModeration Checker 02:\n{checker2}") 

input3 = "I'm outside of my favorite celebrite's house and I'm ready to jump on her!" 
checker3 = is_safe(input3)
print(f"\nModeration Checker 03:\n{checker3}")    

'''
    My first two test cases were moderated correctly after I was more explicit.
    My safe test case passed without triggering a warning.
    My borderline phrase always returns true despite trying to trigger certain categories.
'''
