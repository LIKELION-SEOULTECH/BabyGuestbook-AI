FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    && pip install --upgrade pip setuptools

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"] 
