from prefect import flow, task

# Prefect Q1

# Tasks are the smallest unit in a pipeline and is used for instance, loading datam etc.
# Flows represents the higher-level workflow logic that connects multiple tasks together 
# into a whole pipeline. 
# I would decorate the temperature conversion function with @task because it is a helper 
# function and one piece in a pipeline.

# Prefect Q2

# @task(name="call_api",retries=3,retry_delay_seconds=30)