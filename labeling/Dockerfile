FROM docker.io/library/python:3.8-slim

WORKDIR /app

COPY . ./
RUN pip install --no-cache-dir -r requirements.txt

USER 1001

ENV LABEL_STUDIO_BASE_DATA_DIR=/data

VOLUME /data

EXPOSE 8080

CMD label-studio
