# Bank customer churn prediction


# Table of Contents
- [Description](#description)
- [Tech stack and ML concepts used ](#tech_stack)
- [Local setup](#local_setup)
  - [Prerequisites](#prerequisites)
  - [Set up a virtual environment](#setup_virtual_environment)
  - [Start the web service](#local_setup)
  - [Verify successful setup](#verify_successful_setup)
  - [Sending requests to the web service](#sending_requests)
- [Deployment to the cloud](#deployment)
- [Load testing](#load_testing)
- [Prediction results](#prediction_results)


# Description <a name = "description"></a>
<!---
Write 1-2 paragraphs describing the purpose of your project.
-->
This project trains different machine learning models to predict customer churn at a bank.
The model is served as a web service using AWS Elastic Beanstalk.

To use the churn prediction service, make requests to the web app by sending a JSON
representation of a customer to the `/predict` endpoint with information about a
customer. The churn prediction service will return true or false depending on whether
the service predicts that the customer will churn or not, along with the churn
probability.

The dataset used can be found in Kaggle [here](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data).


# Tech stack and ML concepts used <a name = "tech_stack"></a>
<!---
A list of software used to build the application and ML concepts used.
-->
* Python
* Polars
* Pipenv
* Docker
* Elastic Beanstalk
* Logistic Regression
* Random Forest
* Gradient Boosting (XGBoost)

# Running Locally <a name = "running_locally"></a>


## Prerequisites <a name = "prerequisites"></a>
<!---
A list of things you need to install and how to install them. 
-->
* Python, version 3.11.3
* Pipenv
* [Docker](https://docs.docker.com/engine/install/)
* To deploy the web service to the cloud, you will need an AWS account and also be
    authenticated to run AWS CLI commands ([AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html)).


## Local setup <a name = "local_setup"></a>
<!---
A step by step series of instruction that tells you how to get the application running locally.
-->

1. From your terminal, clone this repository and open the `midterm` directory.

### Set up virtual environment <a name = "setup_virtual_environment"></a>

1. Create a virtual environment and install dependencies:
   ```
   pipenv install
   ```
3. Activate the virtual environment:
   ```
   pipenv shell
   ```
You will now be able to execute cells in the development.ipynb notebook.



### Starting the churn prediction web service <a name = "start_web_app"></a>
1. Build the docker image:
   ```
   docker build -t churn-prediction .
   ```
2. Run the Docker container:
   ```
   docker run -it --rm -p 9696:9696 churn-prediction
   ```

### Verify the setup was successful <a name = "verify_successful_setup"></a>
<!---
Provide an example command to verify that the setup was successful.
-->
In a different terminal window, make a GET request to the `/health` endpoint of the 
churn prediction service: 
```
curl http://localhost:9696/health
```
This should return the following response:
`{"status":"running"}`

### Sending requests to the web service <a name = "sending_requests"></a>
In a different terminal window, you can make POST requests to the web service to predict
of a customer is going to churn.

Here is an example command:
```
curl -X POST -H "Content-Type: application/json" -d @data/example_customer.json http://localhost:9696/predict
```


## 🚀 Deploying to the cloud <a name = "deployment"></a>
<!---
Add additional notes about how to deploy this application to the cloud.
-->
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

See below a screen recording of the deployment to Elastic Beanstalk:


## Prediction Results <a name = "prediction_results"></a>
| Model                      | Validation Set Accuracy | Test Set Accuracy |
|----------------------------|-------------------------|-------------------|
| Logistic Regression        | 67%                     | -                 |
| Random Forest              | 86%                     | -                 |
| Gradient Boosting (XGBoost | 87%                     | 86%               |
