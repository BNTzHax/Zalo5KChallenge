# python /model/infer.py
# !git clone https://github.com/ultralytics/yolov5
cd yolov5
ls /data/private_test/images
python detect.py --weights /model/yolo_fdm_best.pt --source /data/private_test/images/1.jpg --device cpu --save-conf --save-txt --conf-thres 0.3 --img 400 --nosave 
# change cpu to device 0 when submitting