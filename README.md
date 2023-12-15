# AI-breast-cancer 
This repo is the implementation for "**Automated Breast Cancer Detection Mimicking Human Diagnostic Process**"
## Abstract
## Directory
- [Abstract](#abstract)
- [Dataset](#dataset)
- [Code Description](#code-description)
- [Citing & Authors](#citing--authors)
## Dataset
This study utilized the following two datasets:

- **The Chinese Mammography Database (CMMD):**  
  This database was conducted on 1,775 patients from China with benign or malignant breast disease who underwent mammography examination between July 2012 and January 2016. The database consists of 3,728 mammographies from these 1,775 patients, with biopsy-confirmed types of benign or malignant tumors. For 749 of these patients (1,498 mammographies), the database also includes patients' molecular subtypes. Image data were acquired on a GE Senographe DS mammography system. The data can be obtained from this [link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508)

- **Chinese Breast Disease Clinical Imaging Database:**  
  This database includes 176 mammographic images and 84 corresponding patient complaints from 84 female breast disease patients.The data can be obtained from this [link](https://medbooks.ipmph.com/yx/imageLibrary/2578.html)

## Code Description
- `cal_mean_std.py`: Calculates the mean and variance of mammography image datasets.
- `cal_para_quan.py`: Calculates the parameter quantity of models.
- `config.yaml`: A configuration file for setting up the project environment.
- `image_only.py`, `xgboost_muti.py`, `xgboost_text.py`: Decision-makers for text-only, image-only, and combined text and image analysis.
- `load_data.py`: Loads datasets for processing.
- `losses.py`: Computes loss functions for model training.
- `model.py`: Builds various types of models for image processing and analysis.
- `reconstruct.py`: Tests the reconstruction effects of image autoencoder models.
- `test_classifier.py`, `test_vit.py`: Tests various types of image classifiers.
- `train_ae.py`: Trains autoencoder models.
- `train_classifier.py`, `train_vit.py`: Trains various types of image classifiers.
- `utilss.py`: Contains general utility functions for the project.
- `requirement.txt`: The required python packages.
## Installation
/```
/pip install -r requirements.txt
/```
## Usage

## Citing & Authors
