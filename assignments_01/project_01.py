import pandas as pd
from prefect import flow, task
import matplotlib.pyplot as plt 

from prefect.logging import get_run_logger
from scipy.stats import ttest_ind,pearsonr
import seaborn as sns

from pathlib import Path

# Folder containing project_01.py
BASE_DIR = Path(__file__).resolve().parent

# Input and output folders
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"

# Task 1
paths = [ 
        INPUT_DIR/ "world_happiness_2015.csv",
        INPUT_DIR/ "world_happiness_2016.csv",
        INPUT_DIR/ "world_happiness_2017.csv",
        INPUT_DIR/ "world_happiness_2018.csv",
        INPUT_DIR/ "world_happiness_2019.csv",
        INPUT_DIR/ "world_happiness_2020.csv",
        INPUT_DIR/ "world_happiness_2021.csv",
        INPUT_DIR/ "world_happiness_2022.csv",
        INPUT_DIR/ "world_happiness_2023.csv",
        INPUT_DIR/ "world_happiness_2024.csv",
        ]


@task(retries=3,retry_delay_seconds=2)
def happiness_data():
    final_dataframe = []          

    for path in paths:
        read_path = pd.read_csv(path, sep=";",decimal=',')

        # Raw files use commas for decimal values.
        # Convert decimal commas to periods so numeric columns can be used in analysis.
        object_cols = read_path.select_dtypes(include="object").columns
        read_path[object_cols] = read_path[object_cols].replace(",", ".", regex=True)

        # Convert cleaned numeric values to numeric types.
        read_path = read_path.apply(pd.to_numeric, errors="ignore")
        read_path = read_path.rename(
            columns={"Ladder score": "Happiness score"}
        )
        
        #Grab last piece of the filename instead of hardcoding an index
        year = int(path.stem.split("_")[-1])
        final_path = read_path.assign(Year=year)
        final_dataframe.append(final_path)
        
    # merge all info into one csv --> go's to output folder
    happiness_merged = pd.concat(final_dataframe)
    happiness_merged["Year"] = happiness_merged["Year"].astype(int)
    happiness_merged.to_csv(OUTPUT_DIR/"merged_happiness.csv", index=False)
    return happiness_merged

    '''
    Save to the outputs folder inside assignments_01.
    Since this script runs from the assignments_01 directory,
    the correct relative path is "OUTPUT_DIR/...".
    '''
# Task 2
@task(retries=3,retry_delay_seconds=2)
def happy_stats(df):
    # A little cleanup
    df["Happiness score"] = df["Happiness score"].astype(float).round(2)

    happy_score = df["Happiness score"]
    happy_mean = happy_score.mean()
    happy_median = happy_score.median()
    happy_std = happy_score.std()
    
    happy_mean_year = df.groupby("Year")["Happiness score"].mean()
    happy_mean_region = df.groupby("Regional indicator")["Happiness score"].mean()
    logger = get_run_logger() 

    logger.info(f"Mean:\n {happy_mean}")
    logger.info(f"\nMedian:\n {happy_median}")
    logger.info(f"\nStandard deviation:\n {happy_std}")
    
    logger.info(f"\nGrouped mean:\n {happy_mean_year}")
    logger.info(f"\nGrouped mean:\n {happy_mean_region}")

# Task 3
@task(retries=3,retry_delay_seconds=2)
def visuals(df):
    logger =get_run_logger()
    happy = df["Happiness score"]
    
    #Cleanup
    gdp = df["GDP per capita"]
    gdp = gdp.astype(float)
    
    # Histogram
    plt.hist(df["Happiness score"],bins=20,color="red",alpha=0.9)
    plt.title("Happiness Over the Years")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    
    plt.savefig(OUTPUT_DIR/"happiness_histogram.png",dpi=300)
    plt.show()
    logger.info("Saved histogram.")
    
    # Boxplot
    sns.boxplot(data=df, x="Year", y ="Happiness score")
    plt.title("Happiness Distribution by Years")
    plt.savefig(OUTPUT_DIR/"happiness_by_year.png",dpi=300)
    plt.show()
    logger.info("Saved boxplot.")
    
    # Scatter Plot
    fig, ax = plt.subplots()
    ax.set_title("Happiness by GDP")
    ax.scatter(gdp,happy,color="turquoise")
    ax.set_xlabel("GDP")
    ax.set_ylabel("Happy")
    
    plt.savefig(OUTPUT_DIR/"gdp_vs_happiness.png",dpi=300)
    plt.show()
    logger.info("Saved scatter plot.")
    
    # Heatmap
    numeric = df.select_dtypes(include="number")
    heat_corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(12,6))
    sns.heatmap(heat_corr,annot=True,cmap="coolwarm",fmt=".2f")
    
    plt.title("Correlation Heatmap")
    plt.savefig(OUTPUT_DIR/"correlation_heatmap.png",dpi=300)
    plt.show()
    logger.info("Saved heatmap.")

# Task 4
@task(retries=3,retry_delay_seconds=2)
def hypothesis(df):
    year1 = df[df['Year'] == 2019]["Happiness score"]
    year2 = df[df['Year'] == 2020]["Happiness score"]
    year_hypo = ttest_ind(year1,year2)
    logger =get_run_logger() 

    logger.info(f"Statistic: {year_hypo.statistic}\n")
    logger.info(f"Pvalue: {year_hypo.pvalue}\n")
    logger.info(f"Mean 2019: {year1.mean()}\n")
    logger.info(f"Mean 2020: {year2.mean()}")
    
    '''
    Results:
    The p-value is greater than or equal to 0.05, so the difference in
    average happiness scores between 2019 and 2020 is not statistically significant.
    This suggests there is not enough evidence to conclude that average
    global happiness scores changed between 2019 and 2020.
    '''
    #Test of my choice
    country1 = df[df['Country'] == "Switzerland"]["Happiness score"]
    country2 = df[df['Country'] == "United States"]["Happiness score"]
    country_hypo = ttest_ind(country1,country2)
    
    logger.info(f"Statistic: {country_hypo.statistic}\n")
    logger.info(f"Pvalue: {country_hypo.pvalue}\n")
    logger.info(f"Mean Switzerland: {country1.mean()}\n")
    logger.info(f"Mean United States: {country2.mean()}")
    
    if country_hypo.pvalue < 0.05:
        logger.info(
            "The average happiness scores for Switzerland and the United States are significantly different."
        )
    else:
        logger.info(
            "There is no statistically significant difference between Switzerland and the United States."
        )

# Task 5
@task(retries=3,retry_delay_seconds=2)
def pearson_happiness(df):
    
    happy = df["Happiness score"]
    gdp = df["GDP per capita"]
    social_support = df["Social support"] 
    freedom = df["Freedom to make life choices"]    
    corruption = df["Perceptions of corruption"]
    generosity = df["Generosity"]
    
    logger = get_run_logger() 
    logger.info(df[["Year", "Happiness score"]].dtypes)
    
    pearson1 = pearsonr(gdp,happy)
    pearson2 = pearsonr(social_support,happy)
    
    # Make sure bad values are dropped and both columns are the same length
    temp = df[["Healthy life expectancy", "Happiness score"]].dropna()
    pearson3 = pearsonr(temp["Healthy life expectancy"],temp["Happiness score"])
    
    pearson4 = pearsonr(freedom,happy)
    pearson5 = pearsonr(corruption,happy)
    pearson6 = pearsonr(generosity,happy)
    
    
    #List of results
    correlations = [
        ("GDP per capita", pearson1),
        ("Social support", pearson2),
        ("Healthy life expectancy", pearson3),
        ("Freedom to make life choices", pearson4),
        ("Perceptions of corruption", pearson5),
        ("Generosity", pearson6),
    ]
    
    adjusted_alpha = 0.05/len(correlations)
    
    # Loop over correlation name & result
    for name, result in correlations:
        logger.info(f"\n{name}")
        logger.info(f"Correlation (r): {result.statistic:.3f}")
        logger.info(f"P-value: {result.pvalue:.6f}")
        logger.info(f"Adjusted alpha: {adjusted_alpha}")
        
        if result.pvalue < adjusted_alpha:
            logger.info("Significant after Bonferroni correlation: Yes\n")
        else:
            logger.info("Significant after Bonferroni correlation: No\n")

# Task 6
@task(retries=3,retry_delay_seconds=2)
def summary_report(df):
   logger =get_run_logger() 
   logger.info("Merged dataset\n")
   logger.info(f"Number of countries: {df['Country'].nunique()}")
   logger.info(f"Number of years: {df['Year'].nunique()}")
   
   regional_means = (
       df.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
   )
   logger.info(f"\nTop 3 regions by mean happiness:")
   for region, score in regional_means.head(3).items():
       logger.info(f"{region}: {score:.3f}")
       
   logger.info(f"\nBottom 3 regions by mean happiness:")
   for region, score in regional_means.tail(3).items():
       logger.info(f"{region}: {score:.3f}")
       
   mean_by_year = (
       df.groupby("Year")["Happiness score"].mean().sort_values(ascending=False)
   )
   
   logger.info(
    f"Mean happiness was {mean_by_year[2019]:.2f} in 2019 and "
    f"{mean_by_year[2020]:.2f} in 2020. "
    "Because the p-value (0.595) is greater than 0.05, the observed difference "
    "is not statistically significant."
    )
  
   happy = df["Happiness score"]
   social_support = df["Social support"]
   pearson2 = pearsonr(social_support, happy)
   adjusted_alpha = 0.05 / 6

   if pearson2.pvalue < adjusted_alpha:
      logger.info(
            f"""Social support showed the strongest positive correlation with
            happiness score (r = {pearson2.statistic:.3f}) and remained 
            statistically significant after applying the Bonferroni correction."""
        )

@flow(name="pipeline_flow")
def happiness_pipeline():
    df = happiness_data()

    happy_stats(df)
    visuals(df)
    hypothesis(df)
    pearson_happiness(df)
    summary_report(df)

if __name__== "__main__":
    happiness_pipeline()