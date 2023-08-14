FROM python:3.11-slim

WORKDIR /usr/src/app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt 

EXPOSE 8080
CMD ["gunicorn", "app:app", "--workers=1", "--threads=1", "--bind=0.0.0.0:8080"]