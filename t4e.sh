docker build -t t4e_zalo . && docker run  -v /Users/namnguyenthe/Workspace/AIChallenges/zalo/data:/data -v /Users/namnguyenthe/Workspace/AIChallenges/zalo/result:/result t4e_zalo /bin/bash /model/predict.sh