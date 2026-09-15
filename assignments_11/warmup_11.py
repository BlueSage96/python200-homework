from prefect import flow, task
from prefect.logging import get_run_logger

# Prefect Q1

# A task is the smallest unit in a pipeline and is used for instance, loading datam etc.
# Flows represents the higher-level workflow logic that connects multiple tasks together 
# into a whole pipeline. 
# I would not decorate a simple Celsius-to-Fahrenheit helper with @task because it is a small, 
# pure calculation that does not need retries or separate orchestration. A task represents a 
# meaningful unit of work that Prefect should track, while a flow coordinates those tasks.

# Prefect Q2

@task(name="call_api",retries=3, retry_delay_seconds=30)

# Prefect Q3

# I would open the failed transform task run in the Prefect UI and inspect its logs. The logs 
# should show the exception type, error message, traceback, and the point where the transform task failed. 
# Because load_enriched depends on the transform output, it would not run after transform failed.

# Production Q1

# response.raise_for_status() raises an exception for an unsuccessful HTTP response, allowing Prefect to mark 
# the task as failed and apply its retry behavior. Merely printing an error for a 500 response does not stop execution, 
# so downstream tasks could receive missing or malformed data. 

# Using 
#   if response.status_code != 200: 
#       print("Something went wrong") 
# # detects the error, but the pipeline continues running with bad data.
 
# In the if statement, if the API returns a 404 or 500, the pipeline continues with an empty 
# or malformed response. That can lead to corrupted downstream data, which leads to misleading results, or a 
# pipeline that appears successful even though the data is wrong.

# Production Q2

# Using upsert(..., on_conflict="date") makes a retry safe because an existing date is updated rather than duplicated. 
# If the pipeline crashes and restarts after some rows were already loaded, a normal insert could fail on duplicate dates,
# while the upsert can safely process those dates again.

# Production Q3

@task
def log_upsert(enrichment_records: list):
    get_run_logger().info(f"Upserted {len(enrichment_records)} enrichment records.")
    
# Production 04

# Idempotency makes the pipeline safe to rerun because upsert updates records that already
# exist instead of creating duplicates. Incremental processing checks weather_enriched for
# dates that have already been processed and only sends new records through the ML model
# and LLM. Without the incremental check, all 365 records would be processed again every
# time the pipeline runs, wasting time and unnecessary LLM API calls. It could also produce
# different LLM summaries for records that were already successfully enriched.