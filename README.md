## A CNN regression model to estimate buildings height maps using Sentinel-1 SAR and Sentinel-2 MSI time series (IGARSS 2023)

## Description: 
Accurate estimation of building heights is essential for urban planning, infrastructure management, and environmental analysis. In this study, we propose a supervised Multimodal Building Height Regression Network (MBHR-Net) for estimating building heights at 10m spatial resolution using Sentinel-1 (S1) and Sentinel-2 (S2) satellite time series. S1 provides Synthetic Aperture Radar (SAR) data that offers valuable information on building structures, while S2 provides multispectral data that is sensitive to different land cover types, vegetation phenology, and building shadows. Our MBHR-Net aims to extract meaningful features from the S1 and S2 images to learn complex spatio-temporal relationships between image patterns and building heights. The model is trained and tested in 10 cities in the Netherlands. Root Mean Squared Error (RMSE), Intersection over Union (IOU), and R-squared (R2) score metrics are used to evaluate the performance of the model. The preliminary results (3.73m RMSE, 0.95 IoU, 0.61 R 2 ) demonstrate the effectiveness of our deep learning model in accurately estimating building heights, showcasing its potential for urban planning, environmental impact analysis, and other related applications. (https://ieeexplore.ieee.org/abstract/document/10283039 )

## Run
```
### train 
train.py train --weight 'WEIGHT_FILE_NAME.h5'
### evaluate
train.py evaluate --weight 'WEIGHT_FILE_NAME.h5'
```

## If you are using this work, please cite:
```
@INPROCEEDINGS{10283039,
  author={Nascetti, Andrea and Yadav, Ritu and Ban, Yifang},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={A CNN Regression Model to Estimate Buildings Height Maps Using Sentinel-1 SAR and Sentinel-2 MSI Time Series}, 
  year={2023},
  volume={},
  number={},
  pages={2831-2834},
  doi={10.1109/IGARSS52108.2023.10283039}}
```

## Contact Information/ Corresponding Author:
Ritu Yadav (email: er.ritu92@gmail.com)
