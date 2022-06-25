FROM debian:latest
RUN apt-get update && apt-get install python3-pip -y
RUN pip install requirements.txt
COPY tests_performances.py /./tests_performances.py
WORKDIR /.
CMD python3 /./tests_performances.py