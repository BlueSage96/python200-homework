import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- ML/LLM Questions ---

# ML/LLM Q1

# The ML classifier is best at fast, predictable predictions from the 
# structured data. For example, the ML classier can make predictions
# for good_for_running and confidence.

# The LLM is best at language, explanations, and judgment. It can
# be used to make human readable predictions such as, "Classify the sentiment. 
# Reply with exactly one word: positive, negative, or neutral."

# If the LLM was used to make predictions, it is possible that the data
# would unreadable to the model or even corrupted. If the ML was used 
# to make prompts, the data would be unreadable to a human or it gives
# inaccurate statements making it harder to properly train the model.


# ML/LLM Q2

# 1. Converting "2023-07-04" to a day of the week:
# Deterministic code, because the answer follows a fixed calendar rule.

# 2. Classifying a freeform job posting as entry-level, mid-level, or senior:
# An LLM, because the task requires interpreting unstructured language and context.

# 3. Predicting customer churn from 15 numeric features and labeled training data:
# Machine learning, because this is a supervised prediction problem with
# structured features and known labels.

# 4. Normalizing city names such as "NYC" and "New York City":
# Deterministic code, because known variants can be mapped to a canonical value
# consistently without needing prediction.

# 5. Summing a revenue column:
# Deterministic code, because addition has an exact result and requires neither
# machine learning nor language interpretation.


# ML/LLM Q3

# Incremental processing prevents the pipeline from processing records that
# have already been enriched. Without it, rerunning the pipeline could send
# all 365 weather records to the LLM again, creating unnecessary API calls
# and additional cost.

# It could also overwrite existing enrichment results even though the raw
# weather data has not changed. Processing only new records makes the pipeline
# more efficient and helps preserve the correctness of data that has already
# been processed successfully.


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

# The prompt should explicitly require two sentences and give each sentence
# a specific purpose. The first sentence should state whether the day is good
# for running, and the second should explain the recommendation using the
# supplied weather conditions. The existing weather values and classifier
# prediction already provide the LLM with the information it needs to explain
# its reasoning, so an additional reasoning parameter is not required.

# The validation logic would also need to change because it should verify that
# the LLM returned exactly two sentences instead of one. The sentence-count
# check would need to accept two sentences as valid and reject responses that
# contain fewer or more than two sentences.

# Prompt Q2

print(f"Prompt 02\n")
def call_with_retry(client, messages, max_retries=3):
    for retries in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response
        
        except Exception:
            if retries < max_retries - 1:
                time.sleep(2)

    return None

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": """Give me a weather prediction for January 01, 2027 in Raleigh, North Carolina.
Tell me the reasoning for your prediction."""
    }
]

response = call_with_retry(client, messages)

retry = call_with_retry(client,messages,3)
print(retry)

# In a production pipeline, I would use retry logic for temporary API failures,
# such as network errors, timeouts, or short-lived service errors, so one
# temporary failure does not immediately stop the pipeline.