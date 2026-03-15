FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    MODEL_BUNDLE_PATH=/opt/model-bundle \
    MODEL_RUN_ID= \
    REGISTRY_DIR=/app/registry \
    UVICORN_WORKERS=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY main.py /app/main.py
COPY start-serving.sh /app/start-serving.sh
COPY model-bundle /opt/model-bundle
COPY metadata /app/deployment-metadata

RUN chmod +x /app/start-serving.sh

EXPOSE 7860

CMD ["/app/start-serving.sh"]
