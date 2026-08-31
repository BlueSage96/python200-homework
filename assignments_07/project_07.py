# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent
from scipy.stats import pearsonr
from pathlib import Path

import pandas as pd
import os
from dotenv import load_dotenv

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')
    
api_key = os.getenv("OPEN_AI_KEY")
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

MERGED_PATH = Path("../assignments_01/outputs/merged_happiness.csv")
DATA_PATH = Path("resources/happiness_project/")

# Task 1
df = None

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset.

    First attempts to load the merged happiness CSV from
    assignments_01/outputs/merged_happiness.csv. If that file does not
    exist, loads and combines the yearly CSV files from
    assignments/resources/happiness_project/.

    Returns:
        A dictionary containing the shape and column names of the
        loaded dataset, or an error dictionary if no files are found.
    """
    global df

    # Preferred path: Assignment 01 merged dataset
    merged_path = Path("../assignments_01/outputs/merged_happiness.csv")

    # Fallback path: Assignment resources
    data_path = Path("resources/happiness_project/")

    if merged_path.exists():
        df = pd.read_csv(merged_path)

    else:
        happiness_files = [
            file for file in data_path.glob("*.csv")
            if file.name != "merged_happiness.csv"
        ]

        if not happiness_files:
            return {"error": "No happiness files were found."}

        dataframes = [pd.read_csv(file) for file in happiness_files]
        df = pd.concat(dataframes, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }
    
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
     Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary containing descriptive statistics for the column,
        or an error dictionary if the data is not loaded or the column
        does not exist.
    """
    if df is None:
        return { "error": "no data was found"}
    if column not in df.columns:
        return {"error": f"Column '{column}' was not found."}
    return df[column].describe().to_dict()
    
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: First column to correlate.
        col2: Second column to correlate.

    Returns:
        A dictionary containing the column names, Pearson correlation
        coefficient, and p-value, or an error dictionary if the data
        is not loaded or either column is missing.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' was not found."}

    if col2 not in df.columns:
        return {"error": f"Column '{col2}' was not found."}

    result = pearsonr(df[col1], df[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(result.statistic, 4),
        "p_value": round(result.pvalue, 4),
}
    
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> list[dict] | dict:
    """Return the top N countries ranked by a given column for a specific year.
    Args:
      column: The column to rank countries by.
      year: The specific year to filter the dataset by.
      n: The number of top-rated countries to return. Defaults to 5.

    Returns:
        A list of dictionaries containing file's column,
        year, and country, ordered by the requested column's values.
    ...
    """
    if df is None:
        return{"error":"data is not found."}
    
    if column not in df.columns:
        return {"error": "Column was not found."}
    
    if "Year" not in df.columns:
        return {"error": f"{year} was not found in year column."}
    
    if "Country" not in df.columns:
        return {"error":"Country was not found in the 'countries' column."}
    
    filtered_year = df[df["Year"] == year]  
    filtered_year = filtered_year.sort_values(by=column,ascending=False)
    return filtered_year[["Country", column]].head(n).to_dict("records") #get the first 5 countries


# Task 2

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).

For custom plots, DO NOT call load_happiness_data() to get the DataFrame.
load_happiness_data() returns metadata only (shape and columns).
Read the actual merged CSV directly with pandas:
../assignments_01/outputs/merged_happiness.csv
Use the actual rows and values from that CSV.
Never invent, simulate, randomize, or substitute data.

For regional happiness plots, group the actual rows by Year and Regional indicator and
calculate the mean Happiness score for each group. Only create files explicitly requested by the user. 
Do not create additional plots or output files.

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# Task 3
queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the Happiness score column.",
    "What is the correlation between GDP per capita and Happiness score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot Happiness score over the years as a line chart, with one line per Regional indicator. Save the plot to outputs/happiness_by_region.png."
]

# Task 4

# My query 1
my_query_1 = """
            Show me the top 5 least happiest countries in 2018.
            Use the actual data from ../assignments_01/outputs/merged_happiness.csv.
            Sort Happiness score in ascending order and return the five countries with the lowest scores.
            """
# Comment: The agent should use Python/pandas to sort Happiness score
# in ascending order because get_top_n_countries() returns the highest values.

# My query 2

my_query_2 = """
Create a bar chart for Healthy life expectancy vs Happiness score for the top two
(highest happiness scores) and bottom two (lowest happiness scores) countries.

Use the actual data from ../assignments_01/outputs/merged_happiness.csv.
Do not use load_happiness_data() as the DataFrame because it only returns metadata.

Use pandas to:
1. Load the actual CSV.
2. Find the two countries with the highest Happiness score.
3. Find the two countries with the lowest Happiness score.
4. Get the actual Healthy life expectancy and Happiness score values for those four countries.
5. Label the chart with the actual country names.
6. Create the bar chart comparing Healthy life expectancy and Happiness score.
7. Save it to outputs/happiness_life_expectancy.png.

Do not invent, simulate, or substitute any values.
"""

# Comment: Code generation only.


if __name__ == "__main__":
    # Task 3 queries
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(f"Task 3: {response}")
    
    # Task 4 queries
    response_1 = agent.run(my_query_1, reset=False)
    print(f"Task 4 response 01: {response_1}")
    
    response_2 = agent.run(my_query_2, reset=False)
    print(f"Task 4 response 02: {response_2}")
    
# Task 5

# --- Reflection ---

# 1. In Query 3, how did the agent communicate whether the correlation
#    was statistically significant? Did it use the p-value correctly?
#    What threshold did it apply?
#
# The agent used the p-value returned by compute_correlation to determine
# statistical significance. It compared the p-value to 0.05. A p-value
# below 0.05 was treated as statistically significant.


# 2. Did any of the agent's responses surprise you?
#    Describe one specific example.
#
# I was surprised that the agent could use multiple tools in sequence.
# For example, it could load the dataset and then use the loaded data
# to calculate the requested correlation without me manually writing
# the pandas code.


# 3. What one additional tool would make this agent meaningfully more useful?
#
# A plot-saving tool would make the agent more useful. It could accept
# the data, chart type, labels, and output path and reliably save the
# requested chart. This would help with questions that require creating
# and saving visualizations.