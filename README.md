<div align="center">
<h1> ADCNet: Adaptive Dual-Domain Collaborative Network for Remote Sensing Change Detection </h1>
</div>

## 🎈 News

- [2024.11.10] Training and inference code released

## 🚀 Introduction


The challenges: 
(a) In encoding phase, how to effectively integrate spatial- and frequency-domain information while maintaining boundary integrity and enabling global-local modeling; 
(b) In reconstruction phase, how to efficiently aggregate multi-scale features while preserving high resolution.

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="asserts/ADCNet.png?raw=true">
</div>


Illustrates the overall architecture of ADCNet, which mainly consists of Dual-Domain Guided Fusion Module and Adaptive Frequency-Domain Hybrid Attention Module. (a) The proposed Dual-Domain Guided Fusion Module (DGF), (b) the proposed Multi-Dimensional Feature Fusion Module (MDFF), and (c) the proposed Adaptive Frequency Selection Module(AFS).


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n ASCSeg python=3.8
conda activate ASCSeg
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
```

### 2. Prepare Datasets

- Download datasets: LEVIR-CD from this [link](https://justchenhao.github.io/LEVIR), SYSU-CD from this [link](https://gitee.com/fuzhou-university-wq_0/SYSU-CD), and CDD-CD from this [link](https://aistudio.baidu.com/aistudio/datasetdetail/89523).

### 3. Train the ADCNet

```
python train.py --datasets LEVIR-CD
pre-training file is saved to ./checkpoints/CD_MPDNet_LEVIR_b16_lr0.01_train_val_300_linear/best.pth
concrete information see ./ADCNet/train.py, please
```

### 3. Test the ADCNet

```
python test.py --datasets LEVIR-CD
testing results are saved to ./vis folder
concrete information see ./ADCNet/test.py, please
```


## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="asserts/Visualization.png?raw=true">
</div>



## 🎫 License

The content of this project itself is licensed under [LICENSE](LICENSE).
