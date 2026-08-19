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

DATA_PATH = Path("resources/happiness_project/")
MERGED_PATH = DATA_PATH / "merged_happiness.csv"


# Task 1
df = None

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    
    If a merged CSV exists, load that file. Otherwise, load
    each yearly CSV file from the happiness_project directory and combine
    them into one DataFrame.

    Returns:
        A dictionary containing the shape and column names of the
        loaded DataFrame, or an error message if no CSV files are found.
    """
    global df #important since df is created outside of function
    
    # Load merged CSV from DATA_PATH
    if MERGED_PATH.exists():
        df = pd.read_csv(MERGED_PATH)
    else:
        # if file does not exist, fall back to loading & merging
        # all yearly CSVs from DATA_PATH
        happiness_files = []
        for file in DATA_PATH.glob("*.csv"):
            if file.name != "merged_happiness.csv":
                happiness_files.append(file)
            
        if not happiness_files:
            return {"error": "No happiness files were found."}
        
        dataframes = []
        for file in happiness_files:
            happy_df = pd.read_csv(file)
            dataframes.append(happy_df)
            
        df = pd.concat(dataframes,ignore_index=True)
    return {"shape":df.shape, "columns": df.columns.tolist()}

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
For custom plots, use the actual dataset rows from the project's merged happiness CSV.
Do not use the metadata returned by load_happiness_data() as the dataset or invent data.
For regional happiness plots, use Year, Regional indicator, and Happiness score from
the CSV and calculate the mean Happiness score for each Year and Regional indicator.
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
    "Plot Happiness score over the years as a line chart, with one line per Regional indicator. Save the plot to resources/happiness_by_region.png.",
]

for query in queries:
    print(f"\n--- Query: {query} ---")
    response = agent.run(query, reset=False)
    print(response)