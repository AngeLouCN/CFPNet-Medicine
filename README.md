# CFPNet-M: A Light-Weight Encoder-Decoder Based Network for Multimodal Biomedical Image Real-Time Segmentation
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/Drive.jpg" width="500" height="215" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/ISBI.jpg" width="500" height="175" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/Infrared Breast.jpg" width="500" height="125" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/Cvc-ClinicDB.jpg" width="500" height="150" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/ISIC2018.jpg" width="500" height="150" alt="Result"/></div>
This repository contains the implementation of a novel light-weight real-time network (CFPNet-Medicine: CFPNet-M) to segment different types of biomedical images. It is a medical version of CFPNet, and the dataset we used from top to bottom are **DRIVE, ISBI-2012, Infrared Breast, CVC-ClinicDB and ISIC 2018**. The details of CFPNet-M and CFPNet can be found here respectively.  

[CFPNet-M](https://arxiv.org/ftp/arxiv/papers/2105/2105.04075.pdf),
[CFPNet Paper](https://arxiv.org/ftp/arxiv/papers/2103/2103.12212.pdf),
[DC-UNet](https://github.com/AngeLouCN/DC-UNet) and
[CFPNet Code](https://github.com/AngeLouCN/CFPNet)

**:fire: NEWS :fire:**
**The pytorch version is available** [**pytorch-version**](https://github.com/AngeLouCN/CFPNet-Medicine/tree/main/CFPNetM-pytorch).

## Architecture of CFPNet-M
### CFP module
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/cfp module.png" width="750" height="300" alt="Result"/></div>

### CFPNet-M
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/fig 3.jpg" width="400" height="400" alt="Result"/></div>

## Dataset

In this project, we test five datasets:

- [x] Infrared Breast Dataset
- [x] Endoscopy (CVC-ClinicDB)
- [x] Electron Microscopy (ISBI-2012)
- [x] Drive (Digital Retinal Image)
- [x] Dermoscopy (ISIC-2018)

## Usage

### Prerequisities

The following dependencies are needed:

- Kearas == 2.2.4
- Opencv == 3.3.1
- Tensorflow == 1.10.0
- Matplotlib == 3.1.3
- Numpy == 1.19.1

### training

You can download the datasets you want to try, and just run: for **UNet, DC-UNet, MultiResUNet, ICNet, CFPNet-M, ESPNet and ENet**, the code is in the folder ```network```. For **Efficient-b0, MobileNet-v2 and Inception-v3**, the code is in the ```main.py```. Choose the segmentation model you want to test and run:

```
main.py
```

## Segmentation Results of Five datasets

<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/seg_table_1.png" width="700" height="600" alt="Result_table"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/seg_table_2.png" width="675" height="600" alt="Result_table"/></div>

## Speed and FLOPs
The code of calculate FLOPs are in ```main.py```, you can run them after training.
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/speed.png" width="700" height="200" alt="Result_table"/></div>

## Citation
```
@article{lou2023cfpnet,
  title={Cfpnet-m: A light-weight encoder-decoder based network for multimodal biomedical image real-time segmentation},
  author={Lou, Ange and Guan, Shuyue and Loew, Murray},
  journal={Computers in Biology and Medicine},
  pages={106579},
  year={2023},
  publisher={Elsevier}
}

@inproceedings{lou2021cfpnet,
  title={Cfpnet: channel-wise feature pyramid for real-time semantic segmentation},
  author={Lou, Ange and Loew, Murray},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={1894--1898},
  year={2021},
  organization={IEEE}
}

@inproceedings{lou2021dc,
  title={DC-UNet: rethinking the U-Net architecture with dual channel efficient CNN for medical image segmentation},
  author={Lou, Ange and Guan, Shuyue and Loew, Murray H},
  booktitle={Medical Imaging 2021: Image Processing},
  volume={11596},
  pages={115962T},
  year={2021},
  organization={International Society for Optics and Photonics}
}
```

