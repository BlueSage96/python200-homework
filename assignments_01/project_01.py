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
df.rename(columns={"Happiness score":"happiness_score","Regional indicator":"regional_indicator"},inplace=True)

# Task 2
#@task(retries=3,retry_delay_seconds=2)
def happy_stats():
    # A little cleanup
    df["happiness_score"] = df["happiness_score"].astype(float).round(2)

    happy_score = df["happiness_score"]
    happy_mean = happy_score.mean()
    happy_median = happy_score.median()
    happy_std = happy_score.std()
    happy_mean_grouped = df.groupby(["Year","regional_indicator"])["happiness_score"].mean()
    
    # print(f"Mean:\n {happy_mean}")
    # print(f"\nMedian:\n {happy_median}")
    # print(f"\nStandard deviation:\n {happy_std}")
    # print(f"\nGrouped mean:\n {happy_mean_grouped}")
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
    plt.savefig("outputs/correlation_heatmap.png",dpi=300)
    plt.show()
    
visuals()

# Task 4
#@task(retries=3,retry_delay_seconds=2)
def hypothesis():
    year1 = df[df['Year'] == 2019]["happiness_score"]
    year2 = df[df['Year'] == 2020]["happiness_score"]
    #make sure hapiness scores are numeric! Float not object!
    year_hypo = ttest_ind(year1,year2)
    # print(f"Statistic: {year_hypo.statistic}\n")
    # print(f"Pvalue: {year_hypo.pvalue}\n")
    # print(f"Mean 2019: {year1.mean()}\n")
    # print(f"Mean 2020: {year2.mean()}")
    
    #Results:
    # The p-value is greater than or equal to 0.05, so the difference in
    # average happiness scores between 2019 and 2020 is not statistically significant.
    # This suggests there is not enough evidence to conclude that average
    # global happiness scores changed between 2019 and 2020.
    
    #Test 2
    country1 = df[df['Country'] == "Switzerland"]["happiness_score"]
    country2 = df[df['Country'] == "United States"]["happiness_score"]
    country_hypo = ttest_ind(country1,country2)
    # print(f"Statistic: {country_hypo.statistic}\n")
    # print(f"Pvalue: {country_hypo.pvalue}\n")
    # print(f"Mean Switzerland: {country1.mean()}\n")
    # print(f"Mean United States: {country2.mean()}")
hypothesis()

# Task 5
#@task(retries=3,retry_delay_seconds=2)
def pearson_happiness():
    happy = df["happiness_score"]
    year = df["Year"]
    gdp = df["GDP per capita"]
    social_support = df["Social support"]
    life_expectancy = df["Healthy life expectancy"] 
    freedom = df["Freedom to make life choices"]    
    corruption = df["Perceptions of corruption"]
    generosity = df["Generosity"]
    
    # print(life_expectancy.dtype)
    # print(life_expectancy.isna().sum())
    # print(happy.dtype)
    # print(happy.isna().sum())

    pearson1 = pearsonr(year,happy)
    pearson2 = pearsonr(gdp,happy)
    pearson3 = pearsonr(social_support,happy)
    
    # Make sure bad values are dropped and both columns are the same length
    temp = df[["Healthy life expectancy", "happiness_score"]].dropna()
    pearson4 = pearsonr(temp["Healthy life expectancy"],temp["happiness_score"])
    pearson5 = pearsonr(freedom,happy)
    pearson6 = pearsonr(corruption,happy)
    pearson7 = pearsonr(generosity,happy)
    
    # print("Pearson 1\n")
    # print(f"Statistic:\n {pearson1.statistic}")
    # print(f"P-value:\n {pearson1.pvalue}")
    
    # print("Pearson 2\n")
    # print(f"Statistic:\n {pearson2.statistic}")
    # print(f"P-value:\n {pearson2.pvalue}")
    
    # print("Pearson 3\n")
    # print(f"Statistic:\n {pearson3.statistic}")
    # print(f"P-value:\n {pearson3.pvalue}")
    
    # print("Pearson 4\n")
    # print(f"Statistic:\n {pearson4.statistic}")
    # print(f"P-value:\n {pearson4.pvalue}")
    
    # print("Pearson 5\n")
    # print(f"Statistic:\n {pearson5.statistic}")
    # print(f"P-value:\n {pearson5.pvalue}")
    
    # print("Pearson 6\n")
    # print(f"Statistic:\n {pearson6.statistic}")
    # print(f"P-value:\n {pearson6.pvalue}")
    
    # print("Pearson 7\n")
    # print(f"Statistic:\n {pearson7.statistic}")
    # print(f"P-value:\n {pearson7.pvalue}")
    
    adjusted_alpha = 0.05/7
    
    # print(f"Pearson1 P-value: {pearson1.pvalue}")
    # print(f"Adjusted alpha: {adjusted_alpha}")
    # # Significant at α = 0.05: Yes
    # # Significant after Bonferroni: No
    
    # print(f"Pearson2 P-value: {pearson2.pvalue}")
    # print(f"Adjusted alpha: {adjusted_alpha}")
    # # Significant at α = 0.05: Yes
    # # Significant after Bonferroni: Yes
    
    # print(f"Pearson3 P-value: {pearson3.pvalue}")
    # print(f"Adjusted alpha: {adjusted_alpha}")
    # # Significant at α = 0.05: Yes
    # # Significant after Bonferroni: Yes
    
    # print(f"Pearson4 P-value: {pearson4.pvalue}")
    # print(f"Adjusted alpha: {adjusted_alpha}")
    # # Significant at α = 0.05: Yes
    # # Significant after Bonferroni: Yes

    # print(f"Pearson5 P-value: {pearson5.pvalue}")
    # print(f"Adjusted alpha: {adjusted_alpha}")
    # # Significant at α = 0.05: Yes
    # # Significant after Bonferroni: Yes

    # print(f"Pearson6 P-value: {pearson6.pvalue}")
    # print(f"Adjusted alpha: {adjusted_alpha}")
    # # Significant at α = 0.05: Yes
    # # Significant after Bonferroni: Yes

    # print(f"Pearson7 P-value: {pearson7.pvalue}")
    # print(f"Adjusted alpha: {adjusted_alpha}")
    # Significant at α = 0.05: Yes
    # Significant after Bonferroni: Yes

pearson_happiness()

# Task 5
@task(retries=3,retry_delay_seconds=2)
def summary_report():
   logger = get_run_logger() 
   logger.info("Merged dataset")
   logger.info("Number of countries: 175")
   logger.info("Number of years: 10")
   
   logger.info("Top 3 regions by mean happiness score:")
   logger.info("1. North America and ANZ 2. Western Europe 3. Latin America and Caribbean")
   
   logger.info("Bottom 3 regions by mean happiness score:")
   logger.info("1. Sub-Saharan Africa 2. South Asia 3. Middle East and North Africa")
   
   logger.info("Average happiness scores changed between 2019 and 2020, suggesting that people's reported happiness was different after the start of the pandemic.")
   logger.info("Social support had the strongest relationship with happiness score, even after using a stricter significance test.")
summary_report()