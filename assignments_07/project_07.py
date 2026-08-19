# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent
from scipy.stats import pearsonr
from pathlib import Path

import pandas as pd
import os
from dotenv import load_dotenv

api_key = os.getenv("OPEN_AI_KEY")
if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

MERGED_PATH = Path("assignments_01/outputs/merged_happiness.csv")
DATA_PATH = Path("assignments/resources/happiness_project/")

# Task 1
df = None

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    If the merged happiness CSV exists in assignments_01/outputs,
    load that file. Otherwise, load and combine the yearly CSV files
    from the resources/happiness_project directory.

    Returns:
        A dictionary containing the shape and column names of the
        loaded DataFrame, or an error message if no CSV files are found.
    """
    global df

    # Preferred path: merged dataset from Assignment 01
    merged_path = Path("../assignments_01/outputs/merged_happiness.csv")

    # Fallback path: yearly happiness CSV files in Assignment 07
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

        dataframes = []

        for file in happiness_files:
            dataframes.append(pd.read_csv(file))

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
        col1: First column whose value will be correlated.
        col2: Second column whose value will be correlated.
        
    Returns:
        A dictionary containing the two column names, Pearson correlation
        coefficient, and p-value, or an error dictionary if the data or
        columns are invalid.
    """
    
    if df is None:
        return{"error": "No column is found in the CSV loaded."}

    if col1 not in df.columns:
        return{"error": f"{col1} was not found."}

    if col2 not in df.columns:
        return{"error": f"{col2} was not found."}
    
    pearson_r = pearsonr(df[col1],df[col2])
    p_value = pearson_r.pvalue
    stat = pearson_r.statistic
    
    return { "col1": col1, 
                "col2": col2,
                "pearson_r": round(stat,4),
                "p_value" : round(p_value,4)}
        
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
    return filtered_year[["Year","Country",column]].head(n).to_dict("records") #get the first 5 countries


# Task 2
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).

For custom plots, use the actual rows from the project's merged happiness CSV, not the
metadata returned by load_happiness_data(). Never invent, simulate, randomize, or
substitute data. Use the actual column names and values from the CSV.

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
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]

for query in queries:
    print(f"\n--- Query: {query} ---")
    response = agent.run(query, reset=False)
    print(response)
    
# Task 4
# My query 1
my_query_1 = "Show me the top 5 least happiest countries in 2018."   # replace with your question
response_1 = agent.run(my_query_1, reset=False)
print(response_1)
# Comment: Both code use and the get_top_n_countries tool.

# My query 2
my_query_2 = "Create a bar chart for Healthy life expectancy vs Happiness score for the top two (highest happiness scores) and bottom two (lowest happiness scores) countries. Make sure to list each country and not 'Country 1', etc. Save the chart to resources/happiness_life_expectancy.png."   # replace with your question
response_2 = agent.run(my_query_2, reset=False)
print(response_2)
# Comment: Code generation only.


# Task 5

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?

#    It used the compute_corrleation tool and was very accurate p-value. 
#    The threshold used was the "significance" dictionary key to determine whether 
#    or not there was a correlation between GDP per capita and the Happiness score.

# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.

#    For the first question for query 3, the agent failed to get the happiness_data's shape, 
#    so it improvised and got the length of the shape and got a list of keys for the data_columns. 
#    This shows me how creative smolagents are.

# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)

#    A tool that can automatically have the agent: can tell the agent to automatically do simple things 
#    like making sure it gives proper axis and legend labels. The tool could also have the agent 
#    question its decisions before it goes back and check its own output.

if __name__ == "__main__":
    queries,
    my_query_1,
    my_query_2
