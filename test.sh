#!/bin/bash

python3 main.py \
-s veri \
-t veri \
-a seresnet50 \
--resume "/user/HS400/da01075/coursework/CV/vehicle_reid/logs/seresnet_xent+triplet+3augmentations/model.pth.tar" \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--save-dir logs/eval-veri-seresnet_xent+triplet+3augmentations 

