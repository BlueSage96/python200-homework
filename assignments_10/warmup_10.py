import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- ML/LLM Questions ---

# ML/LLM Q1

"""
The ML classifier is best at fast, predictable predictions from the 
structured data. For example, the ML classier can make predictions
for good_for_running and confidence.

The LLM is best at language, explanations, and judgment. It can
be used to make human readable predictions such as, "Classify the sentiment. 
Reply with exactly one word: positive, negative, or neutral."

If the LLM was used to make predictions, it is possible that the data
would unreadable to the model or even corrupted. If the ML was used 
to make prompts, the data would be unreadable to a human or it gives
inaccurate statements making it harder to properly train the model.
"""

# ML/LLM Q2

"""
1. Converting a date string like "2023-07-04" to day-of-week

I would use deterministic code since this is not a prediction or prompt command.

2. Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text

I would use a LLM because reading comprehension is required and judgment that rule-based code handles poorly.

3. Predicting customer churn given 15 numeric features and a labeled training dataset

This situation would be best handled by a ML since there is prediction and labeled training dataset involved.

4. Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form

Data manipulation is involved so using deterministic code is the best move.

5. Summing a column of revenue figures

Data manipulation is best used for this scenario since there is no training nor predictions involved.
"""

# ML/LLM Q3

"""
Incremental processing only classifies records that have not been enriched. If the
data is rewritten, if something went wrong with the LLM during the last run, the data could be corrupted.
If the orignal records are not backed up, that information may be lost forever. 

For this project, there only needs to be 365 records per year. If there is no incremental processing,
those records are repeatedly ran with inaccurate data possibly replacing the original data. As a result,
the weather model would be trained on bad data and will not work as expected.
"""

# --- Prompt Questions ---

# Prompt Q1

SYSTEM_PROMPT = ("""
                You are writing a two-sentence running recommendation for a daily weather summary app.
                You will receive weather conditions for a single day and a machine learning prediction
                about whether the day is good for running.
                Write exactly two sentences — the first sentence needs to state the prediction and the
                second sentence explains the reasoning.
                Do not use bullet points, headers, or phrases like 'Based on the data'.
            """
)

"""The if/else would have to be changed to also include the reasoning for the prediction.
The function needs another parameter (i.e. good_for_running_reason) that's added to the if/else. 
For example: prediction_text = "good for running" if good_for_running else "not ideal for running" becomes: 
prediction_text = "good for running" if good_for_running and good_for_running_reason else "not ideal for running because of {good_for_running_reasons}"""

# Prompt Q2

print(f"Prompt 02\n")
def call_with_retry(client, messages, max_retries=3):
    for retries in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT}, 
                    { "role":"user", "content": messages}
                ],
                max_tokens = max_retries
            )
            return response
        except Exception:
            if retries < max_retries - 1:
                time.sleep(2)

    return None

messages = """Give me a prediction weather for January 01, 2027 in Raleigh, North Carolina. 
Tell me the reasoning for your prediction."""

retry = call_with_retry(client,messages,3)
print(retry)

# In production, retry logic can be useful when an API request fails temporarily
# because of a network issue, timeout, or short-lived service error.