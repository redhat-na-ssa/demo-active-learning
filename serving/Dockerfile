FROM docker.io/library/python:3.8-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# USER 1001

EXPOSE 9090

CMD python _wsgi.py
