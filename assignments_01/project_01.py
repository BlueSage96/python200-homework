import numpy as np
import pandas as pd
from prefect import flow, task

# Task 1
paths = [ 
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2015.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2016.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2017.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2018.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2019.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2020.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2021.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2022.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2023.csv",
         "https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_2024.csv",
        ]


# make function and use @task
# @task(retries=3,retry_delay_seconds=2)
def happiness_data():
    final_dataframe = []          

    for path in paths:
        read_path = pd.read_csv(path, sep=";")
        read_path = read_path.rename(columns={"Ladder score":"Happiness score"})
        loop_path = pd.DataFrame(read_path)
        path_year = path.replace(".csv", "").rsplit("_")

        final_path = loop_path.assign(Year=path_year[3])
        final_dataframe.append(final_path)
        
    # merge all info into one csv --> go's to output folder
    happiness_merged = pd.concat(final_dataframe)
    happiness_merged.to_csv("outputs/merged_happiness.csv",index=False)
happiness_data()

data = pd.read_csv("outputs/merged_happiness.csv")
df = pd.DataFrame(data)

# Task 2
#@task(retries=3,retry_delay_seconds=2)
def happy_stats():
    # A little cleanup
    df.rename(columns={"Happiness score":"happiness_score","Regional indicator":"regional_indicator"},inplace=True)

    df["happiness_score"] = df["happiness_score"].str.replace(",",".")
    df["happiness_score"] = df["happiness_score"].astype(float).round(2)

    happy_score = df["happiness_score"]
    happy_mean = happy_score.mean()
    happy_median = happy_score.median()
    happy_std = happy_score.std()
    
    # close but I need to include happiness_score and .mean() -- look up examples!
    happy_mean_grouped = df.groupby(["Year","regional_indicator"])["happiness_score"].mean()
    
    print(f"Mean:\n {happy_mean}")
    print(f"\nMedian:\n {happy_median}")
    print(f"\nStandard deviation:\n {happy_std}")
    print(f"\nGrouped mean:\n {happy_mean_grouped}")
happy_stats()