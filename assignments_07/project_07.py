# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent
import pandas as pd
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"

# Task 1
df = None

for d in DATA_PATH:
    read_path = pd.read_csv(d)
    if not read_path:
        ALL_Path = "assignments/resources/happiness_project"
        

@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into memory.
    """
    