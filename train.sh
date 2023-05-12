#!/bin/bash
arch_name=seresnet18_concatfusion
save_dir=exp_CNN_ARCHITECTURES/global$arch_name
model_file="/user/HS400/da01075/coursework/CV/vehicle_reid/$save_dir/*/model.pth.tar"

python3 main.py \
--root ./datasets/ \
-s veri \
-t veri \
-a $arch_name \
--train-sampler RandomIdentitySampler \
--height 224 \
--width 224 \
--random-erase \
--optim adam \
--lr 0.0001 \
--max-epoch 60 \
--lr-scheduler multi_step \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--eval-freq 5 \
--save-dir $save_dir \
--resume $model_file



# --random-erase \
# --color-jitter \
# --color-aug \