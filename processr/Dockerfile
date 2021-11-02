FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED 1
ENV DISPLAY=:99
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

# Install apt dependencies
RUN mkdir -p /usr/share/man/man1
RUN apt-get update && apt-get upgrade -y
RUN apt install default-jdk scala git -y

RUN pip3 install newspaper3k
RUN mkdir -p /usr/local/nltk_data
COPY nltk_data  /usr/local/nltk_data

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT python FlaskUI.py
