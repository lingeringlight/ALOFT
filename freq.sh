#!/bin/bash

device=4
data='OfficeHome'


for t in `seq 0 0`
do
  for domain in `seq 3 3`
  do
    if [[ t -gt -1 || (t -eq -1 && domain -gt 0)]];
    then
      python frequency_analyse.py \
      --target $domain \
      --device $device \
      --seed $t \
      --batch-size 64 \
      --data $data \
      --epochs 50 \
      --dataloader_DG_GFNet 0 \
      --data_root '/data/DataSets/' \
      --freq_analyse 1 \
      --freq_analyse_last 1 \
      --resume '/data/gjt/GFNet_results/OfficeHome/adamw6.25e-05E50_dataDG_train_nogray/'
    fi
  done
done


