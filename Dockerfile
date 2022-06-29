FROM debian:latest
COPY ./requirements.txt .
RUN apt-get update && apt-get install python3-pip -y
RUN pip install -r requirements.txt
COPY knc_eval.pkl /./knc_eval.pkl
COPY svm_eval.pkl /./svm_eval.pkl
COPY log_eval.pkl /./log_eval.pkl
COPY dtc_eval.pkl /./dtc_eval.pkl
COPY knc_model.pkl /./knc_model.pkl
COPY svm_model.pkl /./svm_model.pkl
COPY log_model.pkl /./log_model.pkl
COPY dtc_model.pkl /./dtc_model.pkl
COPY fraud.csv /./fraud.csv
COPY api.py /./api.py
COPY pipelines.py /./pipelines.py
WORKDIR /.
EXPOSE 8000
CMD uvicorn api:api --host 0.0.0.0
