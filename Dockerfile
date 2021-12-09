FROM bitnami/pytorch

# RUN mkdir -p /model

# RUN make /app
# WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY infer.py /model/infer.py
COPY predict.sh /model/predict.sh
# COPY keep_data /data/keep_data

VOLUME /data

# CMD ls /

# CMD /bin/bash /model/predict.sh