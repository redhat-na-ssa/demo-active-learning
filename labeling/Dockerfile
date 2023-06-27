# FROM python:3.8-slim
FROM registry.redhat.io/ubi7/python-38:latest

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

ENV LABEL_STUDIO_BASE_DATA_DIR=/opt/app-root/src

VOLUME /data

EXPOSE 8080

CMD label-studio
