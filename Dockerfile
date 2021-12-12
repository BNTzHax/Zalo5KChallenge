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

COPY models/* /model/

COPY cacert.pem /opt/bitnami/python/lib/python3.7/site-packages/certifi/cacert.pem

COPY infer_clf_models.py /model/infer_clf_models.py
COPY train_clf_models.py /model/train_clf_models.py
COPY infer_yolo_fmd.py /app/yolov5/infer_yolo_fmd.py
COPY ensemble_result.py /app/yolov5/ensemble_result.py
COPY utils/general.py /app/utils/general.py
COPY fold_meta/train_resolve_duplicates_relabel_dist_v1_fold_split.csv /fold_meta/train_resolve_duplicates_relabel_dist_v1_fold_split.csv
COPY predict.sh /model/predict.sh
