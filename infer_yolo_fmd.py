import pandas as pd
import os
import imagesize
import shutil
import torch

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
TEST_FOLDER = '/data/private_test'
PREDICTION_FOLDER = '/inter_data/'
if(not os.path.exists(PREDICTION_FOLDER)):
    os.makedirs(PREDICTION_FOLDER)

test_df = pd.read_csv(f'{TEST_FOLDER}/private_test_meta.csv')

w_path = '/model/yolo_fdm_best.pt'
test_path = f'{TEST_FOLDER}/images'
test_files = os.listdir(test_path)

out_dir = f'{PREDICTION_FOLDER}/predict_private_test/'
if(os.path.exists(out_dir)):
    shutil.rmtree(out_dir)

for file in test_files:
    full_path = os.path.join(test_path, file)
    w, h = imagesize.get(full_path)
    file = file.replace('.jpg', '')
    if w >= h:
        selected_edge = w
    else:
        selected_edge = h

    cmd = f'python detect.py --weights {w_path} --source {full_path} \
     --device {DEVICE} --save-conf --save-txt --conf-thres 0.3 \
     --img {selected_edge} --nosave --project {out_dir} \
     --name {file}'

    os.system(cmd)
    # print(cmd)
    