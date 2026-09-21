# Build context is the repository root, so the model trained in stage 2 can be
# copied into the image.
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY code/models/features.py ./features.py
COPY code/deployment/api/main.py ./main.py

# Copied last: this layer changes on every pipeline run.
COPY models/model.pkl models/model_meta.json ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
