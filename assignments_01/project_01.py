import pandas as pd
from prefect import flow, task
import matplotlib.pyplot as plt 

from prefect.logging import get_run_logger
from scipy.stats import ttest_ind,pearsonr
import seaborn as sns

# Task 1
paths = [ 
         "inputs/world_happiness_2015.csv",
         "inputs/world_happiness_2016.csv",
         "inputs/world_happiness_2017.csv",
         "inputs/world_happiness_2018.csv",
         "inputs/world_happiness_2019.csv",
         "inputs/world_happiness_2020.csv",
         "inputs/world_happiness_2021.csv",
         "inputs/world_happiness_2022.csv",
         "inputs/world_happiness_2023.csv",
         "inputs/world_happiness_2024.csv",
        ]


@task(retries=3,retry_delay_seconds=2)
def happiness_data():
    data = pd.read_csv("outputs/merged_happiness.csv")
    df = pd.DataFrame(data)
    
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
        
        #Grab last piece of the filename instead of hardcoding an index
        year = int(path_year[-1])
        final_path = loop_path.assign(Year=year)
        final_dataframe.append(final_path)
        
    # merge all info into one csv --> go's to output folder
    happiness_merged = pd.concat(final_dataframe)
    happiness_merged["Year"] = happiness_merged["Year"].astype(int)
    happiness_merged.to_csv("outputs/merged_happiness.csv",index=False)

    return happiness_merged

# Task 2
@task(retries=3,retry_delay_seconds=2)
def happy_stats(df):
    # A little cleanup
    df["Happiness score"] = df["Happiness score"].astype(float).round(2)

    happy_score = df["Happiness score"]
    happy_mean = happy_score.mean()
    happy_median = happy_score.median()
    happy_std = happy_score.std()
    happy_mean_grouped = df.groupby(["Year","Regional indicator"])["Happiness score"].mean()
    
    logger = get_run_logger() 

    logger.info(f"Mean:\n {happy_mean}")
    logger.info(f"\nMedian:\n {happy_median}")
    logger.info(f"\nStandard deviation:\n {happy_std}")
    logger.info(f"\nGrouped mean:\n {happy_mean_grouped}")

# Task 3
@task(retries=3,retry_delay_seconds=2)
def visuals(df):
    happy = df["Happiness score"]
    years = df["Year"]
    
    #Cleanup
    gdp = df["GDP per capita"]
    gdp = gdp.astype(float)
    
    heat_cols = df[["Happiness score","GDP per capita","Social support",
                       "Healthy life expectancy","Freedom to make life choices",
                       "Generosity", "Perceptions of corruption","Year"]]
    
    # Histogram
    plt.hist(df["Happiness score"],bins=20,color="red",alpha=0.9)
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

# Task 4
@task(retries=3,retry_delay_seconds=2)
def hypothesis(df):
    year1 = df[df['Year'] == 2019]["Happiness score"]
    year2 = df[df['Year'] == 2020]["Happiness score"]
    year_hypo = ttest_ind(year1,year2)
    logger = get_run_logger() 

    logger.info(f"Statistic: {year_hypo.statistic}\n")
    logger.info(f"Pvalue: {year_hypo.pvalue}\n")
    logger.info(f"Mean 2019: {year1.mean()}\n")
    logger.info(f"Mean 2020: {year2.mean()}")
    
    #Results:
    # The p-value is greater than or equal to 0.05, so the difference in
    # average happiness scores between 2019 and 2020 is not statistically significant.
    # This suggests there is not enough evidence to conclude that average
    # global happiness scores changed between 2019 and 2020.
    
    #Test 2
    country1 = df[df['Country'] == "Switzerland"]["Happiness score"]
    country2 = df[df['Country'] == "United States"]["Happiness score"]
    country_hypo = ttest_ind(country1,country2)
    logger.info(f"Statistic: {country_hypo.statistic}\n")
    logger.info(f"Pvalue: {country_hypo.pvalue}\n")
    logger.info(f"Mean Switzerland: {country1.mean()}\n")
    logger.info(f"Mean United States: {country2.mean()}")

# Task 5
@task(retries=3,retry_delay_seconds=2)
def pearson_happiness(df):
    happy = df["Happiness score"]
    year = df["Year"]
    gdp = df["GDP per capita"]
    social_support = df["Social support"]
    life_expectancy = df["Healthy life expectancy"] 
    freedom = df["Freedom to make life choices"]    
    corruption = df["Perceptions of corruption"]
    generosity = df["Generosity"]
    
    logger = get_run_logger() 
    logger.info(df[["Year", "Happiness score"]].dtypes)
    
    pearson1 = pearsonr(year,happy)
    pearson2 = pearsonr(gdp,happy)
    pearson3 = pearsonr(social_support,happy)
    
    # Make sure bad values are dropped and both columns are the same length
    temp = df[["Healthy life expectancy", "Happiness score"]].dropna()
    pearson4 = pearsonr(temp["Healthy life expectancy"],temp["Happiness score"])
    
    pearson5 = pearsonr(freedom,happy)
    pearson6 = pearsonr(corruption,happy)
    pearson7 = pearsonr(generosity,happy)
    
    logger.info("Pearson 1\n")
    logger.info(f"Statistic:\n {pearson1.statistic}")
    logger.info(f"P-value:\n {pearson1.pvalue}")
    
    logger.info("Pearson 2\n")
    logger.info(f"Statistic:\n {pearson2.statistic}")
    logger.info(f"P-value:\n {pearson2.pvalue}")
    
    logger.info("Pearson 3\n")
    logger.info(f"Statistic:\n {pearson3.statistic}")
    logger.info(f"P-value:\n {pearson3.pvalue}")
    
    logger.info("Pearson 4\n")
    logger.info(f"Statistic:\n {pearson4.statistic}")
    logger.info(f"P-value:\n {pearson4.pvalue}")
    
    logger.info("Pearson 5\n")
    logger.info(f"Statistic:\n {pearson5.statistic}")
    logger.info(f"P-value:\n {pearson5.pvalue}")
    
    logger.info("Pearson 6\n")
    logger.info(f"Statistic:\n {pearson6.statistic}")
    logger.info(f"P-value:\n {pearson6.pvalue}")
    
    logger.info("Pearson 7\n")
    logger.info(f"Statistic:\n {pearson7.statistic}")
    logger.info(f"P-value:\n {pearson7.pvalue}")
    
    adjusted_alpha = 0.05/7
    
    logger.info(f"Pearson1 P-value: {pearson1.pvalue}")
    logger.info(f"Adjusted alpha: {adjusted_alpha}")
    # Significant at α = 0.05: Yes
    # Significant after Bonferroni: No
    
    logger.info(f"Pearson2 P-value: {pearson2.pvalue}")
    logger.info(f"Adjusted alpha: {adjusted_alpha}")
    # Significant at α = 0.05: Yes
    # Significant after Bonferroni: Yes
    
    logger.info(f"Pearson3 P-value: {pearson3.pvalue}")
    logger.info(f"Adjusted alpha: {adjusted_alpha}")
    # Significant at α = 0.05: Yes
    # Significant after Bonferroni: Yes
    
    logger.info(f"Pearson4 P-value: {pearson4.pvalue}")
    logger.info(f"Adjusted alpha: {adjusted_alpha}")
    # Significant at α = 0.05: Yes
    # Significant after Bonferroni: Yes

    logger.info(f"Pearson5 P-value: {pearson5.pvalue}")
    logger.info(f"Adjusted alpha: {adjusted_alpha}")
    # Significant at α = 0.05: Yes
    # Significant after Bonferroni: Yes

    logger.info(f"Pearson6 P-value: {pearson6.pvalue}")
    logger.info(f"Adjusted alpha: {adjusted_alpha}")
    # Significant at α = 0.05: Yes
    # Significant after Bonferroni: Yes

    logger.info(f"Pearson7 P-value: {pearson7.pvalue}")
    logger.info(f"Adjusted alpha: {adjusted_alpha}")
    # Significant at α = 0.05: Yes
    # Significant after Bonferroni: Yes

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

@flow(name="pipeline_flow")
def happiness_pipeline():
    df = happiness_data()

    happy_stats(df)
    visuals(df)
    hypothesis(df)
    pearson_happiness(df)
    summary_report()

if __name__== "__main__":
    happiness_pipeline()