FROM python:3.11.3-slim

COPY ./requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

VOLUME ["/var/lib/postgresql/data"]

CMD mlflow server \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_NAME} \
    --artifacts-destination ${MLFLOW_ARTIFACT_URI} \
    --port ${MLFLOW_PORT} \
    --host 0.0.0.0


