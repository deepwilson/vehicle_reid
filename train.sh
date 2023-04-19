#!/bin/bash

python3 main.py \
--root ./datasets/ \
-s veri \
-t veri \
-a seresnet18 \
--height 224 \
--width 224 \
--train-sampler RandomIdentitySampler \
--random-erase \
--color-jitter \
--color-aug \
--optim amsgrad \
--lr 0.0001 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 3 \
--test-batch-size 100 \
--eval-freq 5 \
--save-dir logs/tensorboard 

# --resume "/user/HS400/da01075/coursework/CV/vehicle_reid/logs/seresnext50_xent+htri+3augs+RIS_IMPORTANT/model.pth.tar"
# --train-sampler RandomIdentitySampler \
# seresnet18_xent_htri_3augs_128emb_RIS