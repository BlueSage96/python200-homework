# Cloud concepts 01
# What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The core economic model is that developers "rent" the cloud servers to run their AI models from cloud services including AWS and Azure. The idea is to train the machine learning model using the cloud provider's GPU cluster and then shut down the cloud service when the developer is done.
# ___________________________________________________________________________________________________________________________

# Cloud Concepts 02
# What is the difference between vertical scaling and horizontal scaling? Give a concent example of when you might choose each.

Vertical scaling means upgrading the machine by adding more CPU and RAM, and using a bigger GPU. Horizontal scaling means that more machines are used and the modeling work is split across these machines.

I would use vertical scaling if I am training a single model for work as a single indie developer. I would use vertical scaling if I expanded my company to have a few employees/contractors helping me scale my business.

# Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

# A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

Due to the sudden increase of users after a product launch, a vertical scaling would be faster than renting more machines from a cloud provider.

# A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

Since a single machine needs to be upgraded, vertical scaling makes the most sense especially if the model is for a small project.

# A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

Horizontal scaling is the best solution due to the increased demand and it can be scaled again in the future when 100,000+ files per run are needed.

# ___________________________________________________________________________________________________________________________

# Cloud Concepts 03

# Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

# Gmail 

SaaS because Google has built and runs the apps while users do not have to do any setup.

# Azure Virtual Machines

Iaas because Azure allows user to install the software, configure the enviornment, and handle security updates.

# AWS S3 (Simple Storage Service)

IaaS because the service is an object storage service offering scalability, data availability, security, and performance.

# GitHub Codespaces

PaaS because GitHub is a service that provides tools to the users to build with their own code.

# Snowflake

SaaS because Snowflake builds, runs, and maintains its services.

# Supabase

PaaS because Supabase manages the infrastructure and allows users to use their own code.

# Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

IaaS is a flexible system that allows developers to use an app to install, build, and maintain their own projects.
SaaS allows users to use an app or service that is prebuilt for them.
PaaS the provider manages the service while the developer adds their own tools to the service.

# ___________________________________________________________________________________________________________________________

# Cloud Concepts 04
# What is a mananged platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

Managed platforms like Databricks and Snowflake build their platforms directly off of cloud providers like AWS and GCP. What is gained is that the managed platforms manages the cloud resources on our behalf. The downside is less flexibility and potential higher costs than using cloud providers directly.

# ___________________________________________________________________________________________________________________________

# Cloud Concepts 05
# The lesson names two situations where the cloud is probably not the right choice. What are they?

-There is a steep learning curve for cloud computing even for basic use.
-Customer support can be slow, resulting in delays for assistance. AI platforms may not always be helpful since cloud platforms quickly makes changes the AI has not caught up with.

# ___________________________________________________________________________________________________________________________

# Cloud Landscape Question 01
# Name the three hyperscalers. For each, write one sentence describing its primary strengteh and the type of organization most likely to use it.

Hyperscalers:

Amazon Web Services (AWS):

The primary strength of AWS is its broad range of services including EC2 (compute), S3 (object storage), RDS (managed databases), SageMaker (ML platform), Lambda (serverless functions).

Organzations using it: Anthropic, Netflix, Airbnb, etc.

Google Cloud Platform (GCP):

GCP is strongest in data and machine learning with Google building many of the foundational ideas in modern distributed systems (MapReduce, Bigtable, Dremel — the precursor to BigQuery).

Organizations using it: Snapchat, Etsy, Shopify, Mars, etc.

Microsoft Azure:

Microsoft is strongest provider for enterprise and government settings mainly because of its deep integration with Windows, Active Directory and Microsoft 365.

Organization using it: OpenAI, Walmart, Samsung, Fedex, UPS, etc.

# ___________________________________________________________________________________________________________________________

# Cloud Landscape 02

# The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

-Access: Weird Azure setup made it harder for students to join the CTD organization, resulting in tenant-level configuration problems and students being blocked for days.

-Pedagogical fit: Azure Blob Storage stored data in a weird format that was organized by path and not easy to work with.

-Pipeling coherence: The ETL pipeline was not easy to work with in Azure, and Supabase gives the two tables (raw zone and enriched zone) a clear relationship making it easier to inspect and debug the table contents.

# Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

I need to do deep research before choosing a cloud service and determine how steep of a learning curve these cloud services have before building anything. Also, I need to look at how each service saves data and how tables and other properties work with one another for each service.