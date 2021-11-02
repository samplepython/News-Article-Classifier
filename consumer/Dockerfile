FROM python:3.8-slim
WORKDIR /app

ENV PYTHONUNBUFFERED 1
ENV DISPLAY=:99

RUN mkdir -p /usr/share/man/man1

RUN apt update
RUN apt install default-jdk scala git -y

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app
