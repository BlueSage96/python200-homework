import numpy as np
import pandas as pd

from prefect import flow, task
import matplotlib.pyplot as plt 
from prefect.logging import get_run_logger

from scipy.stats import ttest_ind, kstest, pearsonr
from scipy import stats
import seaborn as sns

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


# @task(retries=3,retry_delay_seconds=2)
def happiness_data():
    final_dataframe = []          

    for path in paths:
        read_path = pd.read_csv(path, sep=";")
        #Replace commas with periods
        object_cols = read_path.select_dtypes(include="object").columns
        read_path[object_cols] = read_path[object_cols].replace(",", ".", regex=True)
        read_path = read_path.apply(pd.to_numeric,errors="ignore")
        
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

    df["happiness_score"] = df["happiness_score"].astype(float).round(2)

    happy_score = df["happiness_score"]
    happy_mean = happy_score.mean()
    happy_median = happy_score.median()
    happy_std = happy_score.std()
    happy_mean_grouped = df.groupby(["Year","regional_indicator"])["happiness_score"].mean()
    
    print(f"Mean:\n {happy_mean}")
    print(f"\nMedian:\n {happy_median}")
    print(f"\nStandard deviation:\n {happy_std}")
    print(f"\nGrouped mean:\n {happy_mean_grouped}")
happy_stats()

# Task 3
#@task(retries=3,retry_delay_seconds=2)
def visuals():
    happy = df["happiness_score"]
    years = df["Year"]
    
    #Cleanup
    gdp = df["GDP per capita"]
    gdp = gdp.astype(float)
    
    heat_cols = df[["happiness_score","GDP per capita","Social support",
                       "Healthy life expectancy","Freedom to make life choices",
                       "Generosity", "Perceptions of corruption","Year"]]
    
    # Histogram
    plt.hist(df["happiness_score"],bins=20,color="red",alpha=0.9)
    plt.title("Happiness Over the Years")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig("outputs/happiness_histogram.png",dpi=300)
    plt.show()
    
    # Boxplot
    sns.boxplot(x = years, y = happy,data=df)
    plt.title("Happiness Distribution by Years")
    plt.savefig("outputs/happiness_by_year.png",dpi=300)
    plt.show()
    
    # Scatter Plot
    fig, ax = plt.subplots()
    ax.set_title("Happiness by GDP")
    ax.scatter(gdp,happy,color="turquoise")
    ax.set_xlabel("GDP")
    ax.set_ylabel("Happy")
    plt.savefig("outputs/gdp_vs_happiness.png",dpi=300)
    plt.show()

    # Heatmap
    heat_corr = heat_cols.corr(numeric_only=True)
    plt.figure(figsize=(12,6))
    sns.heatmap(heat_corr,annot=True,cmap="coolwarm",fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png",dpi=300)
    plt.show()
    
visuals()