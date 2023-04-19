#!/bin/bash

python3 main.py \
--root ./datasets/ \
-s veri \
-t veri \
-a seresnet50 \
--height 224 \
--width 224 \
--random-erase \
--color-jitter \
--color-aug \
--optim amsgrad \
--lr 0.0001 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 16 \
--test-batch-size 100 \
--eval-freq 5 \
--save-dir logs/tensor \
--resume "/user/HS400/da01075/coursework/CV/vehicle_reid/logs/seresnet_xent+triplet+3augmentations/model.pth.tar"
