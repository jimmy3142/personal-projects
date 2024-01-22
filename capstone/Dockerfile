FROM python:3.11.3-slim

RUN apt-get update \
    && apt-get install gcc curl -y \
    && apt-get clean \
    && pip install -U pip \
    && pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

WORKDIR capstone

COPY ["app/predict.py", "./app/"]
COPY ["schemas/credit_card_transaction.py", "./schemas/"]

ENTRYPOINT [ "uvicorn", "app.predict:app", "--host=0.0.0.0" ]