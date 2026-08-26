# Cloud concepts 01

# What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The core economic model is that you pay for cloud resources as you use them instead of buying, owning, and maintaining your own servers. You can also shut resources down when you are done using them so you aren’t paying for them anymore.

# ___________________________________________________________________________________________________________________________

# Cloud Concepts 02

# What is the difference between vertical scaling and horizontal scaling? Give a concent example of when you might choose each.

Vertical scaling means making one machine bigger by adding more CPU, RAM, or a better GPU. Horizontal scaling means adding more machines and splitting the workload between them.

I would use vertical scaling when a single machine needs more power. I would use horizontal scaling when I need multiple machines to handle a larger workload.

# Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

# A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

Horizontal scaling would make the most sense because more machines can handle the sudden increase in users.

# A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

Vertical scaling makes the most sense because the machine needs a faster GPU and more RAM.

# A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

Horizontal scaling makes the most sense because the work can be split across multiple machines.

# ___________________________________________________________________________________________________________________________

# Cloud Concepts 03

# Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

# Gmail

SaaS because Google manages the software and users just use the application.

# Azure Virtual Machines

IaaS because users manage the operating system, software, and environment running on the virtual machine.

# AWS S3

IaaS because it provides cloud infrastructure for storing and managing files without requiring users to own the physical storage hardware.

# GitHub Codespaces

PaaS because GitHub manages the development environment while developers work on their own code.

# Snowflake

SaaS because Snowflake manages the software and infrastructure while users access the service.

# Supabase

PaaS because Supabase manages the infrastructure and provides tools for developers to build applications.

# IaaS, PaaS, and SaaS

IaaS gives developers more control over the infrastructure, but they have more things to manage. Azure Virtual Machines is an example.

PaaS manages more of the infrastructure for the developer so they can focus on building their application. Supabase is an example.

SaaS is a finished application that users can access without managing the underlying infrastructure. Gmail is an example.

# ___________________________________________________________________________________________________________________________

# Cloud Concepts 04

# What is a mananged platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

Managed platforms like Databricks and Snowflake handle more of the cloud infrastructure for us, making them easier to use than working directly with AWS or GCP. We gain convenience and simpler management, but give up some flexibility and may pay more.

# ___________________________________________________________________________________________________________________________

# Cloud Concepts 05

# The lesson names two situations where the cloud is probably not the right choice. What are they?

-Cloud may not be the right choice when the project has requirements that are better handled on local infrastructure.
-Cloud may also not be worth it when the cost or complexity is greater than the benefits.

# ___________________________________________________________________________________________________________________________

# Cloud Landscape Question 01

# Name the three hyperscalers. For each, write one sentence describing its primary strengteh and the type of organization most likely to use it.

Hyperscalers:

Amazon Web Services (AWS)

AWS’s primary strength is its huge range of cloud services, so it is useful for organizations that need a lot of different cloud tools.

Organzations using it: Anthropic, Netflix, Airbnb, etc.

Google Cloud Platform (GCP)

GCP is strongest in data and machine learning, making it a good choice for organizations working heavily with data and AI.

Organizations using it: Snapchat, Etsy, Shopify, Mars, etc.

Microsoft Azure

Azure is strongest for enterprise and government organizations because of its integration with Microsoft products and services.

Organizations using it: OpenAI, Walmart, Samsung, Fedex, UPS, etc.


# ___________________________________________________________________________________________________________________________

# Cloud Landscape 02

# The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

-Access: Azure’s setup made it harder for students to join the CTD organization and caused tenant configuration problems.

-Pedagogical fit: Azure Blob Storage made the data harder to work with because of how it organized the files.

-Pipeline coherence: The ETL pipeline was harder to inspect and debug in Azure, while Supabase made the relationship between the raw and enriched tables clearer.

# Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

I should research the tools before starting a project and look at their learning curve, data organization, and how well they fit the project.

# ___________________________________________________________________________________________________________________________

# Cloud Landscape 03

# For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

# You need to store 10 TB of image files and retrieve them by filename from any machine.

Category: Object storage 
Provider: AWS S3
It stores files by key, so it works for storing and retrieving the image files.

# You need to run an ML training job on a GPU for four hours, then shut it down.

Category: ML platform
Provider: AWS SageMaker
It can run an ML training job on a GPU and then be shut down when the job is finished.

# You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.

Category: Serverless compute
Provider: AWS Lambda
It can run the web API without managing servers and automatically scales with demand.

# You need to send structured data to a large language model and get a text response back.

Category: LLM API
Provider: AWS Bedrock
It provides access to large language models so structured data can be sent in and a text response returned.

# ___________________________________________________________________________________________________________________________

# Cloud Landscape 04

# The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

If I made an AI model for game development, I could use AWS SageMaker for ML, Azure OpenAI for the LLM, and GCP for cloud storage. Consolidating everything with one provider could make the project easier to manage, but I could give up features or tools that another provider does better.

My stack would be: Game model --> GCP storage --> LLM (Azure) --> SageMaker (ML)