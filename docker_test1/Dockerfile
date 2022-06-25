FROM debian:latest
RUN apt-get update && apt-get install python3-pip -y
RUN pip install requirements.txt
COPY tests_fraud.py /./tests_fraud.py
WORKDIR /.
CMD python3 /./tests_fraud.py