import numpy as np
import pandas as pd
from prefect import flow, task

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr,name="values")

@task
def clean_data(series):
    return series.dropna()
    
@task    
def summarize_data(series):
    series = ({
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    })
    return series

@flow
def pipeline_flow():
    created = create_series(arr)
    cleaned = clean_data(created)
    summary = summarize_data(cleaned)
    return summary

if __name__ == "__main__":
    pipeline_flow()

# 1. Prefect would add unnecessary orchestration overhead for this small
#    pipeline because the tasks are simple and execute quickly.
#
# 2. Prefect would be useful for larger real-world workflows that need
#    scheduling, retries, notifications, and coordination with cloud
#    services, databases, or other infrastructure.