FROM python:3.13.12-slim

RUN pip install mlflow boto3 pymysql cryptography

ADD . /app
WORKDIR /app