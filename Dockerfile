FROM debian:latest
RUN apt-get update && apt-get install python3-pip -y
RUN pip install requirements.txt
COPY api.py /./api.py
COPY pipelines.py /./pipelines.py
WORKDIR /.

EXPOSE 8000
CMD uvicorn api:api --host 0.0.0.0
