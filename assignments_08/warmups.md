# Cloud concepts 01
# What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The core economic model is that developers "rent" the cloud servers to run their AI models from cloud services including AWS and Azure. The idea is to train the machine learning model using the cloud provider's GPU cluster and then shut down the cloud service when the developer is done.

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