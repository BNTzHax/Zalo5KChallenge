# docker build -t t4e_zalo . 

# infer
docker run --shm-size=8gb -v /home/namntse05438/AIChallenges/Zalo5K/data/:/data \
-v /home/namntse05438/AIChallenges/Zalo5K/Zalo5KChallenge/result:/result \
 t4e_zalo /bin/bash /model/predict.sh

# train
docker run --shm-size=8gb -v /home/namntse05438/AIChallenges/Zalo5K/data/:/data -v /home/namntse05438/AIChallenges/Zalo5K/Zalo5KChallenge/result:/result  -v /home/namntse05438/AIChallenges/Zalo5K/Zalo5KChallenge/trained_models:/trained_models t4e_zalo python /model/train_clf_models.py
