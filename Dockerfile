FROM python:3.8.12-slim-buster

RUN mkdir /srv/application
RUN mkdir /srv/application/src
RUN mkdir /srv/application/config

COPY requirements.txt /srv/application
RUN pip install -r /srv/application/requirements.txt

COPY main.py /srv/application
COPY src  /srv/application/src
COPY config  /srv/application/config

WORKDIR /srv/application

CMD ["python", "main.py"]