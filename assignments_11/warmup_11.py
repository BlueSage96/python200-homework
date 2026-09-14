from prefect import flow, task

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