# Model API image.
# Build context is the repository root, so that the trained model produced by
# stage 2 can be copied into the image.
FROM python:3.11-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Feature engineering shared with the training stage.
COPY code/models/features.py ./features.py

# API code.
COPY code/deployment/api/main.py ./main.py

# Trained model packaged in stage 2 (copied last: this layer changes every run).
COPY models/model.pkl models/model_meta.json ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
