

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
