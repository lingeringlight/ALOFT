## ALOFT: A Lightweight MLP-like Architecture with Dynamic Low-Frequency Transform for Domain Generalization

### Requirements

* Python == 3.7.3
* torch >= 1.8
* Cuda == 10.1
* Torchvision == 0.4.2
* timm == 0.4.12

### Evaluation

To evaluate the performance of the models, you can download the models trained  on PACS as below:

|     Methods     | Acc (%) |                            models                            |
| :-------------: | :-----: | :----------------------------------------------------------: |
| Strong Baseline |  87.76  | [download](https://drive.google.com/drive/folders/1DJfGRSpFPmm1FD-sZRZK3ZObOE_-7Aaq?usp=share_link) |
|     ALOFT-S     |  90.88  | [download](https://drive.google.com/drive/folders/1r2HXwe1O54GfQ9R3H-wL2xyR36YAqcpN?usp=share_link) |
|     ALOFT-E     |  91.58  | [download](https://drive.google.com/drive/folders/1K80RPvOyw25bnAd5EGothqMTBL-YDCdm?usp=share_link) |

Please set the `--eval = 1` and `--resume` as the saved path of the downloaded models.  *e.g.*,  `/trained/model/path/photo/checkpoint.pth`. Then you can simple run:

```
python main_gfnet.py --target $domain --data 'PACS' --device $device --eval 1 --resume '/trained/model/path/photo/checkpoint.pth'
```

### Training

Firstly download the GFNet-H-Ti model pretrained on ImageNet from [here](https://drive.google.com/file/d/1_xrfC7c_ccZnVicYDnrViOA_T1N-xoHI/view?usp=sharing) and save it to `/pretrained_model`. To run ALOFT-E, you could run the following code. Please set the `--data_root` argument needs to be changed according to your folder. 

```
bash ALOFT-E.sh
```

You can also train the ALOFT-S model by running the following code:

```
base ALOFT-S.sh
```

### Acknowledgement
Part of our code is borrowed from the following repositories.
* [GFNet](https://github.com/raoyongming/GFNet): "Global Filter Networks for Image Classification", NeurIPS 2021
* [DSU](https://github.com/lixiaotong97/DSU): "Uncertainty modeling for out-of-distribution generalization", ICLR 2022

We thank to the authors for releasing their codes. Please also consider citing their works.
