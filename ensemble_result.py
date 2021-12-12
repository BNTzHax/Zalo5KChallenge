import pandas as pd
import os
from tqdm import tqdm

TEST_FOLDER = '/data/private_test'

PREDICTION_FOLDER = '/inter_data/'
if(not os.path.exists(PREDICTION_FOLDER)):
    os.makedirs(PREDICTION_FOLDER)

predict_dir = f'{PREDICTION_FOLDER}/predict_private_test/'
predict_list = os.listdir(predict_dir)

clf_mask_df = pd.read_csv(f'{PREDICTION_FOLDER}/clf_predicted_probs.csv')

threshold = 0.5
pred_arr = {}
for predict in tqdm(predict_list):
    full_path = os.path.join(predict_dir, f'{predict}/labels/{predict}.txt')
    if not os.path.isfile(full_path):
        continue
    with open(full_path, 'r') as f:
            full_content = f.read()
            content_arr = full_content.split('\n')
            flag = True
            for bbox in content_arr:
                if bbox != '':
                    bbox_pred = bbox.split(' ')
                    if bbox_pred[0] == '0' and float(bbox_pred[-1]) >= threshold:
                        pred_arr[predict] = 0
                        flag = False
                        break
            if flag == True:
                pred_arr[predict] = 1

ids, mask = [], []
for d_key, d_value in pred_arr.items():
    ids.append(d_key)
    mask.append(d_value)

check_csv = pd.DataFrame()
check_csv['image_id'] = ids
check_csv['check_mask'] = mask
check_csv['image_id']=check_csv['image_id'].astype(int)
merged_csv = clf_mask_df.merge(check_csv, on='image_id', how='left').fillna(1)

pred = merged_csv['check_mask'].values

ens = []
for i, row in merged_csv.iterrows():
    value = row['check_mask'] and row['mask']
    ens.append(int(value))

merged_csv['mask'] = ens

merged_csv['5K'] = merged_csv['mask'] & merged_csv['distancing']

merged_csv[['image_id', 'fname', '5K']].to_csv('/result/submission.csv', index=False)