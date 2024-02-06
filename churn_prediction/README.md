# Bank customer churn prediction


## Table of Contents
- [Description](#description)
- [Tech stack and ML concepts used](#tech_stack)
- [Project structure](#project_structure)
- [Local setup](#local_setup)
  - [Prerequisites](#prerequisites)
  - [Set up a virtual environment](#setup_virtual_environment)
- [Exploring the data](#explore_the_data)
- [Running the web service locally](#running_locally)
  - [Start the web service](#start_web_service)
  - [Verify successful setup](#verify_successful_setup)
  - [Making requests to the web service](#making_requests)
- [Deploying to the cloud](#cloud_deployment)
- [Prediction results](#prediction_results)
- [Future improvements](#future_improvements)


⚠️ The instructions in this project assumes MacOS is the operating system used 
(sorry Windows users!)


## Description <a name = "description"></a>
This project explores different machine learning models to predict customer churn at a
bank. Hyperparameters are tuned on the final churn prediction model and the model is
served as a web service using AWS Elastic Beanstalk.

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
* Ruff formatter and linter
* Elastic Beanstalk
* Logistic Regression
* Random Forest
* Gradient Boosting (XGBoost)


## Project structure <a name = "project_structure"></a>

```shell
├── app/
│    ├── predict.py
│    └── streamlit.py
├── data/
│    ├── bank_customer_churn.csv
│    └── example_customer.json
├── tests/
│    ├── test_predict.py
├── training/
│    ├── train.py
│    ├── xgboost_model.bin
├── .gitignore
├── development.ipynb
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── README.md
```

* **app/:**
  * **predict.py:** Python script to serve the final churn prediction model as a Flask app.
  * **streamlit.py:** Python script to serve the final churn prediction model as a
                          Streamlit app.
* **data/:**
  * **bank_customer_churn.csv**: CSV file with data on bank customer churn.
  * **example_customer.json**: JSON file with data for an example customer. This can be
* **tests/:**
* **training/:** 
  * **train.py**: A python script to train the final XGBoost model and save as a binary file.
  * **xgboost_model.bin:**: The Binary file containing the final churn prediction model.
* **.gitignore:** Contains a list of files and directories to be ignored by git.
* **development.ipynb:** Notebook for interactive development. This includes data
preparation, exploratory data analysis, exploring different ML models, and finally, fine
tuning the parameters and evaluating the performance of each model.
* **Dockerfile:** Defines a Docker image that bundles the project's dependencies and the
final churn prediction model. It also sets up a Flask app that runs on port 9696.
* **Pipfile:** Manages Python dependencies.
* **Pipfile.lock:** Automatically generated to track exact versions of dependencies.
* **README.md:** Provides project information and usage instructions.
  used to make a request to the web service.

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
3. Install the git hook scripts for pre-commit:
   ```
   pre-commit install
   ```

## 🔍 Exploring the data <a name = "exploring_the_data"></a>

With the virtual environment now setup, you will now be able to run the
development.ipynb notebook. This notebook contains exploratory data analysis, trains
multiple ML models, tunes the parameters and evaluates performance metrics, and finally
selects the final model with the best performance.

## 💻 Running the web service locally <a name = "running_locally"></a>


### Start the web service <a name = "start_web_service"></a>

1. Build the docker image:
   ```
   docker build -t churn-prediction .
   ```
2. Run the Docker container:
   ```
   docker run -it --rm -p 9696:9696 churn-prediction
   ```
    
   &ast;*The Docker container can be terminated at any point using the command `control + C`*

### Verify successful setup <a name = "verify_successful_setup"></a>
In a different terminal window, make a GET request to the `/health` endpoint of the 
churn prediction service: 
```
curl http://localhost:9696/health
```
This should return the following response if the endpoint is ready to accept requests:
`{"status": "healthy!"}`


### Making requests to the web service <a name = "making_requests"></a>
In a different terminal window, you can make HTTP requests to the web service using the
POST method to get predictions about whether a customer is likely to churn.

Here is an example command:
```
curl -X POST -H "Content-Type: application/json" -d @data/example_customer.json http://localhost:9696/predict
```

&ast; *the `example_customer.json` file contains an JSON string with details of a single
customer.*

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


## 🥁 ML Predictions results <a name = "prediction_results"></a>
| Model                      | Validation Set Accuracy | Test Set Accuracy |
|----------------------------|-------------------------|-------------------|
| Logistic Regression        | 67%                     | -                 |
| Random Forest              | 86%                     | -                 |
| Gradient Boosting (XGBoost | 87%                     | 86%               |


## ✨  Future improvements <a name = "future_improvements"></a>
* Perform cross validation to check how robust the model is
* Apply feature engineering to try and further improve the model performance.
* Add unit tests to train.py
* Provision infra using Terraform
* Create a Streamlit frontend that users can enter their information in and see their
    churn prediction results.
