import numpy as np
import pandas as pd
from prefect import flow, task

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr,name="values")

@task
def clean_data(series):
    cleaned = create_series(series).dropna()
    return cleaned

@task    
def summarize_data(series):
    series = ({
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    })
    return series

@flow(name="pipeline_flow")
def data_pipeline(arr):
    created = create_series(arr)
    return summarize_data(clean_data(created))

if __name__ == "__main__":
    data_pipeline(arr)
    
# 1. The compute time would become sluggish should the data becomes more complex.

# 2. Prefect can be used for automation tasks: 

# add Slack and email notifications
# use cloud providers such as AWS, GCP, Azure, or Snowflake
# integrate with Docker or Kubernetes infrastructure