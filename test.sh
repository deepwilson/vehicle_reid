#!/bin/bash

python3 main.py \
-s veri \
-t veri \
-a resnet50 \
--resume "/userd/HS400/da01075/coursework/CV/vehicle_reid/logs/resnet50-veri-only_arcface/model.pth.tar-30" \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--save-dir logs/eval-veri 

