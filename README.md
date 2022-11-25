## ALOFT: A Lightweight MLP-like Architecture with Dynamic Low-Frequency Transform for Domain Generalization

### Requirements

* Python == 3.7.3
* torch >= 1.8
* Cuda == 10.1
* Torchvision == 0.4.2
* timm == 0.4.12

### Evaluation

To evaluate the performance of the models, you can download the models trained  on PACS as below:

|     Methods     | Acc(%) |                            models                            |
| :-------------: | :----: | :----------------------------------------------------------: |
| Strong Baseline | 87.76  | [download](https://drive.google.com/drive/folders/1TuJ5hhghykk6HDUn6Jp3oIigOZH6Z8f7?usp=sharing) |
|     ALOFT-S     | 90.88  | [download](https://drive.google.com/drive/folders/1cC6VNKB97EgYdn4M0uYfv0ollByWXj59?usp=sharing) |
|     ALOFT-E     | 91.58  | [download](https://drive.google.com/drive/folders/1GJyKxjX3_q6hS2dzqiCRAwXMITx4BwiB?usp=sharing) |

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