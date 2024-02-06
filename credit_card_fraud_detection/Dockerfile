FROM python:3.11.3-slim

RUN apt-get update \
    && apt-get install gcc curl -y \
    && apt-get clean \
    && pip install -U pip \
    && pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

ENV PYTHONPATH=.

WORKDIR capstone

COPY ["app/predict.py", "./app/"]
COPY ["schemas", "./schemas/"]
COPY ["ml_models", "./ml_models/"]

ENTRYPOINT [ "python", "./app/predict.py" ]