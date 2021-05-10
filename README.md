# CFPNet-M: A Light-Weight Encoder-Decoder Based Network for Multimodal Biomedical Image Real-Time Segmentation
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/Drive.jpg" width="500" height="215" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/ISBI.jpg" width="500" height="175" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/Infrared Breast.jpg" width="500" height="125" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/Cvc-ClinicDB.jpg" width="500" height="150" alt="Result"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/ISIC2018.jpg" width="500" height="150" alt="Result"/></div>
This repository contains the implementation of a novel light-weight real-time network (CFPNet-Medicine: CFPNet-M) to segment different types of biomedical images. It is a medical version of CFPNet, and the dataset we used from top to bottom are DRIVE, ISBI-2012, Infrared Breast, CVC-ClinicDB and ISIC 2018. The details of CFPNet-M and CFPNet can be found here respectively.

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

You can download the datasets you want to try, and just run: for UNet, DC-UNet, MultiResUNet, ICNet, CFPNet-M, ESPNet and ENet, the code is in the folder ```network```. For Efficient-b0, MobileNet-v2 and Inception-v3, the code is in the ```main.py```. Choose the segmentation model you want to test and run:

```
main.py
```

## Segmentation Results of Five datasets

<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/seg_table_1.png" width="700" height="600" alt="Result_table"/></div>
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/seg_table_2.png" width="675" height="600" alt="Result_table"/></div>

## Speed and FLOPs
The code of test speed and FLOPs are in ```main.py```, you can run them after training.
<div align=center><img src="https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/results/seg_table_1.png" width="700" height="600" alt="Result_table"/></div>
