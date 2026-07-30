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


#@task(retries=3,retry_delay_seconds=2)
def happiness_data():
    final_dataframe = []          

    for path in paths:
        read_path = pd.read_csv(path, sep=";")
        #Replace commas with periods
        object_cols = read_path.select_dtypes(include="object").columns
        read_path[object_cols] = read_path[object_cols].replace(",", ".", regex=True)
        read_path = read_path.apply(pd.to_numeric,errors="ignore")
        read_path = read_path.rename(columns={"Ladder score":"Happiness score"})
        
        #Grab last piece of the filename instead of hardcoding an index
        year = int(path.stem.split("_")[-1])
        final_path = read_path.assign(Year=year)
        final_dataframe.append(final_path)
        
    # merge all info into one csv --> go's to output folder
    happiness_merged = pd.concat(final_dataframe)
    happiness_merged["Year"] = happiness_merged["Year"].astype(int)
    
    # Save to the outputs folder inside assignments_01.
    # Since this script runs from the assignments_01 directory,
    # the correct relative path is "OUTPUT_DIR/...".
    happiness_merged.to_csv(OUTPUT_DIR/"merged_happiness.csv", index=False)
    return happiness_merged

# Task 2
#@task(retries=3,retry_delay_seconds=2)
def happy_stats(df):
    # A little cleanup
    df["Happiness score"] = df["Happiness score"].astype(float).round(2)

    happy_score = df["Happiness score"]
    happy_mean = happy_score.mean()
    happy_median = happy_score.median()
    happy_std = happy_score.std()
    happy_mean_grouped = df.groupby(["Year","Regional indicator"])["Happiness score"].mean()
    
    #logger =get_run_logger() 

    print(f"Mean:\n {happy_mean}")
    print(f"\nMedian:\n {happy_median}")
    print(f"\nStandard deviation:\n {happy_std}")
    print(f"\nGrouped mean:\n {happy_mean_grouped}")

# Task 3
#@task(retries=3,retry_delay_seconds=2)
def visuals(df):
    #logger =get_run_logger()
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
    
    plt.savefig(OUTPUT_DIR/"happiness_histogram.png",dpi=300)
    plt.show()
    print("Saved histogram.")
    
    # Boxplot
    sns.boxplot(data=df, x="Year", y ="Happiness score")
    plt.title("Happiness Distribution by Years")
    plt.savefig(OUTPUT_DIR/"happiness_by_year.png",dpi=300)
    plt.show()
    print("Saved boxplot.")
    
    # Scatter Plot
    fig, ax = plt.subplots()
    ax.set_title("Happiness by GDP")
    ax.scatter(gdp,happy,color="turquoise")
    ax.set_xlabel("GDP")
    ax.set_ylabel("Happy")
    
    plt.savefig(OUTPUT_DIR/"gdp_vs_happiness.png",dpi=300)
    plt.show()
    print("Saved scatter plot.")
    
    # Heatmap
    numeric = df.select_dtypes(include="number")
    heat_corr = numeric.corr(numeric_only=True)
    plt.figure(figsize=(12,6))
    sns.heatmap(heat_corr,annot=True,cmap="coolwarm",fmt=".2f")
    
    plt.title("Correlation Heatmap")
    plt.savefig(OUTPUT_DIR/"correlation_heatmap.png",dpi=300)
    plt.show()
    print("Saved heatmap.")

# Task 4
#@task(retries=3,retry_delay_seconds=2)
def hypothesis(df):
    year1 = df[df['Year'] == 2019]["Happiness score"]
    year2 = df[df['Year'] == 2020]["Happiness score"]
    year_hypo = ttest_ind(year1,year2)
    #logger =get_run_logger() 

    print(f"Statistic: {year_hypo.statistic}\n")
    print(f"Pvalue: {year_hypo.pvalue}\n")
    print(f"Mean 2019: {year1.mean()}\n")
    print(f"Mean 2020: {year2.mean()}")
    
    #Results:
    # The p-value is greater than or equal to 0.05, so the difference in
    # average happiness scores between 2019 and 2020 is not statistically significant.
    # This suggests there is not enough evidence to conclude that average
    # global happiness scores changed between 2019 and 2020.
    
    #Test of my choice
    country1 = df[df['Country'] == "Switzerland"]["Happiness score"]
    country2 = df[df['Country'] == "United States"]["Happiness score"]
    country_hypo = ttest_ind(country1,country2)
    
    print(f"Statistic: {country_hypo.statistic}\n")
    print(f"Pvalue: {country_hypo.pvalue}\n")
    print(f"Mean Switzerland: {country1.mean()}\n")
    print(f"Mean United States: {country2.mean()}")
    
    if country_hypo.pvalue < 0.05:
        print(
            "The average happiness scores for Switzerland and the United States are significantly different."
        )
    else:
        print(
            "There is no statistically significant difference between Switzerland and the United States."
        )

# Task 5
#@task(retries=3,retry_delay_seconds=2)
def pearson_happiness(df):
    
    happy = df["Happiness score"]
    gdp = df["GDP per capita"]
    social_support = df["Social support"] 
    freedom = df["Freedom to make life choices"]    
    corruption = df["Perceptions of corruption"]
    generosity = df["Generosity"]
    
    #logger =get_run_logger() 
    print(df[["Year", "Happiness score"]].dtypes)
    
    pearson1 = pearsonr(gdp,happy)
    pearson2 = pearsonr(social_support,happy)
    
    # Make sure bad values are dropped and both columns are the same length
    temp = df[["Healthy life expectancy", "Happiness score"]].dropna()
    pearson3 = pearsonr(temp["Healthy life expectancy"],temp["Happiness score"])
    
    pearson4 = pearsonr(freedom,happy)
    pearson5 = pearsonr(corruption,happy)
    pearson6 = pearsonr(generosity,happy)
    
    adjusted_alpha = 0.05/7
    #List of results
    correlations = [
        ("GDP per capita", pearson1),
        ("Social support", pearson2),
        ("Healthy life expectancy", pearson3),
        ("Freedom to make life choices", pearson4),
        ("Perceptions of corruption", pearson5),
        ("Generosity", pearson6),
    ]
    
    # Loop over correlation name & result
    for name, result in correlations:
        print(f"\n{name}")
        print(f"Correlation (r): {result.statistic:.3f}")
        print(f"P-value: {result.pvalue:.6f}")
        print(f"Adjusted alpha: {adjusted_alpha}")
        
        if result.pvalue < adjusted_alpha:
            print("Significant after Bonferroni correlation: Yes\n")
        else:
            print("Significant after Bonferroni correlation: No\n")

# Task 5
#@task(retries=3,retry_delay_seconds=2)
def summary_report(df):
   #logger =get_run_logger() 
   print("Merged dataset\n")
   print(f"Number of countries: {df['Country'].nunique()}")
   print(f"Number of years: {df['Year'].nunique()}")
   
   regional_means = (
       df.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
   )
   print(f"\nTop 3 regions by mean happiness:")
   for region, score in regional_means.head(3).items():
       print(f"{region}: {score:.3f}")
       
   print(f"\nBottom 3 regions by mean happiness:")
   for region, score in regional_means.tail(3).items():
       print(f"{region}: {score:.3f}")
       
   mean_by_year = (
       df.groupby("Year")["Happiness score"].mean().sort_values(ascending=False)
   )
   
   print(
    f"Mean happiness was {mean_by_year[2019]:.2f} in 2019 and "
    f"{mean_by_year[2020]:.2f} in 2020. "
    "Because the p-value (0.595) is greater than 0.05, the observed difference "
    "is not statistically significant."
    )
  
   print(
    "Social support showed the strongest positive correlation with happiness score and remained statistically significant after applying the Bonferroni correction."
    )

#@flow(name="pipeline_flow")
def happiness_pipeline():
    df = happiness_data()

    happy_stats(df)
    visuals(df)
    hypothesis(df)
    pearson_happiness(df)
    summary_report(df)

if __name__== "__main__":
    happiness_pipeline()