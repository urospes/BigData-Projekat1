FROM bde2020/spark-python-template:3.1.2-hadoop3.2

COPY app.py /app/
COPY requirements.txt /app/

ENV SPARK_APPLICATION_PYTHON_LOCATION /app/app.py
ENV SPARK_APPLICATION_ARGS "hdfs://namenode:9000/input hdfs://namenode:9000/output 23.728415 37.983736 0.3 50 100"
