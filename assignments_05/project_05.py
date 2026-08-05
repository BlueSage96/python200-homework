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

    Return only a JSON array. Do not include explanation, summary, markdown fences, or any text outside the JSON.
    Each item must be an object with exactly two keys:
    - "original": the original bullet
    - "improved": the rewritten bullet

    Bullet points:
    ```
    {bullet_text}
    ```
    """
    print("Bullets:", bullet_text)
    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages,temperature=0)

    result = json.loads(response)
    return result

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

new_bullets = rewrite_bullets(bullets)

print(f"\nTask 02:\n")
print(f"Bullet Point Rewriter:\n{new_bullets}")

# Both the original and updated bullets print out clearly.
# The improvements are meaningful.

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
        
input1 = "I love my cat and I want to hold him!"
checker = is_safe(input1)
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

# Task 05
print(f"\nTask 05:\n")

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": "You are required to maintain a professional tone while talking to users."}
    ]
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=1.0
    )
    
    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()
        print("Input:",user_input)
        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
                    
            new_raw_bullets = rewrite_bullets(raw_bullets)
            print(f"Bullet Point Rewriter:\n{new_raw_bullets}")
            
        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            letter = generate_cover_letter(job_title,background)
            print(f"My cover letter:\n{letter}")
        # 7. Otherwise, handle it as a regular chat turn
        else:
            # - Append the user's message to `messages`
            messages.append({
                "role": "user",
                "content":user_input
                })
            # - Call get_completion(messages)
            response = get_completion(messages,temperature=0)
            # - Print the reply
            print(f"\nUser responses:\n{response}")
            # - Append the reply to `messages` as an assistant message
            messages.append({
                "role": "assistant",
                "content": response
            })
            pass

if __name__ == "__main__":
    run_chatbot()
 
# Task 6  
# Comment block 
'''
Q1. The chatbot's knowledge may be biased towards the IT and Video Game industries 
    since I provided examples related to those industries. The bot may lack knowledge 
    for other professions as a consequence.

Q2. There could be spelling and grammar issues along specific bot "language" that
    employers will recognize. These issues could prevent people from getting jobs and 
    they may not understand what "bot language" is and how to check for it before 
    submitting job applications.
    
Q3. I would consider guardrails such as a UI warning in red, moderation filters that
    do not display inappropriate langauage in the chat, resume or cover letter, and a 
    usage policy that would ban users from continuing to use the job application helper 
    app if they continue using foul language.
'''
