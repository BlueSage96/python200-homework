from prefect import flow, task
from prefect.logging import get_run_logger

# Prefect Q1

# Tasks are the smallest unit in a pipeline and is used for instance, loading datam etc.
# Flows represents the higher-level workflow logic that connects multiple tasks together 
# into a whole pipeline. 
# I would decorate the temperature conversion function with @task because it is a helper 
# function and one piece in a pipeline.

# Prefect Q2

# @task(name="call_api",retries=3,retry_delay_seconds=30)

# Prefect Q3

# The error appears beside of the task (i.e. Task run 'create_series-8ab' - Finished in state Failed()). 
# Then, there a log connected to the task that describes the error. For example, "NameError: name 'arr' is not defined
# 20:52:51.479... Finished in state Failed("Flow run encountered an exception: NameError: name 'arr' is not defined"...)"

# Production Q1

# Using raise_for_statues() allows Prefect to catch the exception and marks the task as "Failed". 
# Using 
#   if response.status_code != 200: 
#       print("Something went wrong") 
# catches the error, but the pipeline continues running with bad data.
 
# In the if statement, if the API returns a 404 or 500, the pipeline continues with an empty 
# or malformed response. That can lead to corrupted downstream data, which leads to misleading results, or a 
# pipeline that appears successful even though the data is wrong.

# Production Q2

# Re-running either load task updates existing rows in place rather than failing or duplicating them. 
# The database always ends up in a consistent state regardless of how many times the load runs. 


# Production Q3

@task
def records(enrichment_records) -> list:
    get_run_logger.info("The number of enriched records are:", enrichment_records)
    
# Production 04

# Using upsert with on_conflict="date" in the load_raw task makes it idempotent. This means re-running 
# the pipeline does not duplicate rows. This task must complete before the transform task runs. 
# Lastly, if the incremental processing check in the transform task was removed, when the pipeline runs 
# the ML and LLM steps, all the 365 records will be overwritten comprimising the data correctness and 
# increases the cost because all of the records are ran each time.