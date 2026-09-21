ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

COPY code/deployment/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/app/app.py ./app.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
