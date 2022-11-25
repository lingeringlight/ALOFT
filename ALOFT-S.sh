#!/bin/bash

device=0
data='PACS'

for t in `seq 0 2`
do
  for domain in `seq 0 3`
  do
      python main_gfnet.py \
      --target $domain \
      --device $device \
      --seed $t \
      --batch-size 64 \
      --data $data \
      --epochs 50 \
      --lr 0.0005 \
      --data_root '/data/DataSets/' \
      --noise_mode 1 \
      --uncertainty_model 1 \
      --uncertainty_factor 0.9 \
      --mask_radio 0.5 \
      --eval 0 \
      --resume '' \
  done
done


