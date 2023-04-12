#!/bin/bash

python3 main.py \
--root ./datasets/ \
-s veri \
-t veri \
-a seresnet50 \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0001 \
--max-epoch 30 \
--stepsize 10 20 \
--train-batch-size 32 \
--test-batch-size 100 \
--eval-freq 5 \
--save-dir logs/seresnet_xent+triplet \
# --resume "/user/HS400/da01075/coursework/CV/vehicle_reid/logs/baseline_xent+triplet+SE/model.pth.tar-5"
