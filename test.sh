#!/bin/bash

python3 main.py \
-s veri \
-t veri \
-a resnet18 \
--resume "/user/HS400/da01075/coursework/CV/vehicle_reid/exp_CNN_ARCHITECTURES/resnet18/Experiment_ARCHresnet18_OPTadam_SHEDULERmulti_step_BS_64_AUGREA+CJ+FLIP_SAMPLERRandomIdentitySampler_HW224224/model.pth.tar" \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--save-dir logs/eval-veri \
--visualize-ranks

