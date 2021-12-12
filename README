# 1. Introduction
- This README is the brief guide on how to use Train4Ever team's docker image to run inference and training

# 2. Run inference
````
    docker run --shm-size=8gb -v [data-folder]:/data \
                -v [result-folder]:/result \
                t4e_zalo /bin/bash /model/predict.sh
````
[data-folder]: Path to folder containing private test data. The structure should be:
- path-to-data
- ....|--------- private_test
- ....................|---------- images/
- ....................|---------- private_test_meta.csv

[result-folder]: Path to result folder


# 3. Run training
````
    docker run --shm-size=8gb -v [data-folder]:/data \
                -v [save-weight-folder]:/trained_models 
                t4e_zalo python /model/train_clf_models.py
````
[data-folder]: Path to folder containing training data. The structure should be:
- path-to-data
- ....|--------- train
- .................|---------- images/
- .................|---------- train_meta.csv
[save-weight-folder]: Path to folder where trained weights will be saved
