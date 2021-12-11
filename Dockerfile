FROM bitnami/pytorch
# FROM rwightman/pytorch-opencv

COPY requirements.txt .
RUN pip install -r requirements.txt

# VOLUME /data
# VOLUME /result

USER root
RUN apt-get -y update
RUN apt-get -y install git
# RUN yes Y | apt-get install libfreetype6-dev

RUN git clone https://github.com/ultralytics/yolov5.git

# Install system libraries required by OpenCV.
RUN apt-get update \
 && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Install OpenCV from PyPI.
RUN pip install opencv-python==4.5.1.48

# RUN apt-get install libfreetype6-dev
RUN yes Y | pip uninstall pillow
RUN pip install --no-cache-dir pillow

COPY infer.py /model/infer.py
COPY predict.sh /model/predict.sh

COPY models/* /model/

COPY cacert.pem /opt/bitnami/python/lib/python3.7/site-packages/certifi/cacert.pem
