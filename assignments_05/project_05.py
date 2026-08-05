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

# Task 01
messages_task1 = [
    {"role":"system","content":
      """
     You are an experienced job application coach who specializes in helping career 
     changers improve their resumes, cover letters, and other job application materials.

     Provide professional, constructive, and encouraging feedback that helps the user 
     present their skills and experience clearly.

     Stay focused only on job application materials. If the user changes the subject, 
     politely redirect the conversation back to resumes, cover letters, or other 
     application documents.

     Never invent work experience, education, certifications, skills, or achievements 
     that the user has not provided.

     Always remind the user to review and edit your suggestions before submitting any 
     job application.

     You may not know the expectations of every industry or employer, so remind the user 
     to use their own judgment and tailor the final document to their specific field.
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
I deliberately included the instruction "Never invent work experience, education,
certifications, skills, or achievements that the user has not provided." I chose
this constraint because inaccurate or fabricated information on a resume or cover
letter could mislead employers and hurt a job applicant's credibility. This
guardrail helps ensure the assistant improves the user's writing while keeping all
content truthful and based only on information the user actually provides.
'''

# Task 02
print("\nTask 02:\n")
print("Bullet Point Rewriter:\n")

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Respond ONLY with valid JSON.
    Do not include markdown.
    Do not include explanations.

    Do not include extra text.
    Return a JSON array where every object contains ONLY these keys:

    "original": "the original bullet",
    "improved": "the rewritten bullet"

    Bullet points:
    ```
    {bullet_text}
    ```
    """
    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages,temperature=0)

    result = json.loads(response)
    
    for bullet in result:
        print(f"Original: {bullet['original']:<55} | Improved: {bullet['improved']}")
    return result

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

new_bullets = rewrite_bullets(bullets)

# Both the original and updated bullets print out clearly.
# The improvements are meaningful.

# Task 03
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    
    Do not copy the examples.
    Use them only as examples of tone and structure.
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

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

cover = generate_cover_letter(job_title,background)

print(f"\nTask 03:\n")
print(f"Cover Letter Generator:\n{cover}")

'''
   I chose examples from two different career changes to show the model how to connect previous 
   professional experience to a new technical role. Both examples demonstrate a confident, specific 
   writing style without exaggerating qualifications, which encouraged the model to write a personalized 
   cover letter that matched the provided job title and background while avoiding invented credentials.
'''

# Task 04
print(f"\nTask 04:\n")

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if not flagged:
        return True
    
    else:
        print("I'm sorry, but I can't help with that request.")
        print("Please rephrase it in a respectful and safe way.")
        return False
        
input1 = "I love my cat and I want to hold him!"
checker = is_safe(input1)
print("Safe input:")
print(checker)      

# Prints AFTER print statement in function!
input2 = "I want to kill my professor! She's terrible at her job!"
checker2 = is_safe(input2)
print("\nFlagged input:")
print(checker2) 

input3 = "I'm outside of my favorite celebrite's house and I'm ready to jump on her!" 
checker3 = is_safe(input3)
print("\nBorderline input:")
print(checker3)  

'''
    The safe test case passed the moderation check without being flagged. 
    The harmful test case was correctly flagged, and the program displayed a message 
    asking the user to rephrase their request respectfully. I also tested a borderline 
    example to see how the moderation model would respond, and it showed that not every 
    concerning statement is automatically flagged, depending on the context and the model's 
    safety criteria.
'''

# Task 05
print(f"\nTask 05:\n")

def run_chatbot():
    messages = [
        {
            "role": "system",
            "content": """
You are a professional job application coach helping career changers.
Help users improve resumes, cover letters, and other job application materials.
Stay focused on job application topics.
Do not invent qualifications or work experience.
Always remind the user to review and edit your suggestions before submitting an application.
If you do not know industry-specific expectations, tell the user to use their own judgment.
"""
        }
    ]

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

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        messages.append({
            "role": "user",
            "content": user_input
        })

        # Resume bullet rewriter
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

            assistant_reply = "Here are your improved resume bullet points:\n\n"

            assistant_reply = "Here are your revised resume bullet points:\n\n"

            for bullet in new_raw_bullets:
                assistant_reply += (
                    f"Original : {bullet['original']}\n"
                    f"Improved: {bullet['improved']}\n\n"
                )

            print(assistant_reply)

            messages.append({
                "role": "assistant",
                "content": assistant_reply
            })
        # Cover letter generator
        elif "cover letter" in user_input.lower():

            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            messages.append({
                "role": "user",
                "content": f"Job title: {job_title}\nBackground: {background}"
            })
            letter = generate_cover_letter(job_title, background)
            print(f"\nJob Application Helper:\n{letter}")

            messages.append({
                "role": "assistant",
                "content": letter
            })

        # Regular chat
        else:

            response = get_completion(messages)

            print(f"\nJob Application Helper:\n{response}")

            messages.append({
                "role": "assistant",
                "content": response
            })
if __name__ == "__main__":
    run_chatbot()
 
# Task 06  
# Comment block 

'''
Format Chosen: Written Reflection

Q1. The chatbot's knowledge may be biased toward the IT and video game industries because
    the examples I provided focused on those fields. As a result, it may generate stronger
    suggestions for technical careers than for professions in healthcare, education, or
    other industries. Users should review the output carefully and adapt it to their own
    profession and experience.

Q2. Users may rely too heavily on the chatbot's suggestions without proofreading them.
    AI-generated resumes and cover letters can contain awkward wording, incorrect grammar,
    or phrases that hiring managers recognize as AI-generated. Incorrect or exaggerated
    information could also hurt a user's chances of getting an interview if it is submitted
    without being reviewed.

Q3. I would include guardrails such as a clear warning reminding users to review and edit
    every response before submitting it. I would also use moderation filters to block
    harmful or inappropriate language, prevent the chatbot from inventing qualifications,
    and display a usage policy explaining acceptable use of the application.
'''