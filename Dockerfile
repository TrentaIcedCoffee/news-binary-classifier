FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /usr/src/app

COPY . .
RUN pip install --no-cache-dir -r requirements_linux_cpu.txt 

EXPOSE 8080
CMD ["gunicorn", "app:app", "--workers=1", "--threads=1", "--bind=0.0.0.0:8080"]