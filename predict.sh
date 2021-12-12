python /model/infer_clf_models.py

cd yolov5
python infer_yolo_fmd.py 

python ensemble_result.py


# change cpu to device 0 when submitting