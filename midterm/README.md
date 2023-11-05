# Bank customer churn prediction


## Table of Contents
- [Description](#description)
- [Tech stack and ML concepts used ](#tech_stack)
- [Local setup](#local_setup)
  - [Prerequisites](#prerequisites)
  - [Set up a virtual environment](#setup_virtual_environment)
- [Running the web service locally](#running_locally)
  - [Start the web service](#start_web_service)
  - [Verify successful setup](#verify_successful_setup)
  - [Making requests to the web service](#making_requests)
- [Deploying to the cloud](#cloud_deployment)
- [Prediction results](#prediction_results)


## Description <a name = "description"></a>
This project explores different machine learning models to predict customer churn at a
bank. The parameters on best performing models are tuned and the final churn prediction
model is served as a web service using AWS Elastic Beanstalk.

The goal of this project is to provide a service to the bank to help understand which
customers are likely to leave, so that they can send promotional offers to encourage
them to stay.

**Using the service**

The churn prediction web service exposes an endpoint, `/predict`, which accepts requests
formatted as JSON. These requests will contain information about a customer. The web
service will respond with true or false to indicate if the customer is predicted to
churn or not, along with the churn probability.

The dataset used can be found in Kaggle [here](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data).


## Tech stack and ML concepts used <a name = "tech_stack"></a>
* Python
* Polars
* Pipenv
* Docker
* Elastic Beanstalk
* Logistic Regression
* Random Forest
* Gradient Boosting (XGBoost)


## Local setup <a name = "local_setup"></a>

### Prerequisites <a name = "prerequisites"></a>
* Python, version 3.11.3
* Pipenv
* [Docker](https://docs.docker.com/engine/install/)
* To deploy the web service to the cloud, you will need an AWS account and also be
    authenticated to run AWS CLI commands ([AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html)).

From your terminal, clone this repository and open the `midterm` directory.

### Set up virtual environment <a name = "setup_virtual_environment"></a>
1. Create a virtual environment and install dependencies:
   ```
   pipenv install
   ```
2. Activate the virtual environment:
   ```
   pipenv shell
   ```
You will now be able to execute cells in the development.ipynb notebook.


## Running the web service locally <a name = "running_locally"></a>

### Start the web service <a name = "start_web_service"></a>

1. Build the docker image:
   ```
   docker build -t churn-prediction .
   ```
2. Run the Docker container:
   ```
   docker run -it --rm -p 9696:9696 churn-prediction
   ```

### Verify successful setup <a name = "verify_successful_setup"></a>
In a different terminal window, make a GET request to the `/health` endpoint of the 
churn prediction service: 
```
curl http://localhost:9696/health
```
This should return the following response:
`{"status":"running"}`

### Making requests to the web service <a name = "making_requests"></a>
In a different terminal window, you can make POST requests to the web service to predict
of a customer is going to churn.

Here is an example command:
```
curl -X POST -H "Content-Type: application/json" -d @data/example_customer.json http://localhost:9696/predict
```


## 🚀  Deploying to the cloud <a name = "cloud_deployment"></a>
We will deploy our web service using Elastic Beanstalk in AWS. See instructions below.

&ast; *Note, in the interest of keeping costs to a minimum, there is a screen recording showing how
deployment of this service works.*

1. Initialise the directory with the Elastic Beanstalk CLI:
   ```
   eb init --platform docker churn-prediction -r eu-west-1
   ```

2. Create an Elastic Beanstalk environment:
   ```
   eb create churn-prediction-env
   ```

3. Make a request to the web service running on Elastic Beanstalk:
   ```
   curl -X POST -H "Content-Type: application/json" -d @data/example_customer.json http://<environment_domain>/predict
   ```
    &ast; change `<environment_domain>`, to the domain name of the EB environment you
    have deployed. You can find this in the AWS console or stdout in the terminal after
    you have created the environment.

The EB environment can be terminated with the following command:
```
eb terminate churn-prediction-env
```

Here is a screen recording showing the deployment to Elastic Beanstalk:

https://github.com/jimmy3142/ml-zoomcamp-2023-projects/assets/65279092/c6d79b68-61ec-4611-b40e-d2eaf254f725



## Prediction Results <a name = "prediction_results"></a>
| Model                      | Validation Set Accuracy | Test Set Accuracy |
|----------------------------|-------------------------|-------------------|
| Logistic Regression        | 67%                     | -                 |
| Random Forest              | 86%                     | -                 |
| Gradient Boosting (XGBoost | 87%                     | 86%               |
