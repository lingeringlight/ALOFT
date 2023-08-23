#!/bin/bash

device=0
data='PACS'

for t in `seq 0 4`
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
      --uncertainty_model 2 \
      --uncertainty_factor 1.0 \
      --mask_radio 0.5 \
      --eval 0 \
      --resume ''
  done
done


