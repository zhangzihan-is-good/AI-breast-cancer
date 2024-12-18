# AI-breast-cancer 
This repo is the implementation for "**MutiBCD: A Multimodal model that simulates the human diagnostic process for automated Breast Cancer Detection**"
## Abstract
To enhance the accuracy of BC detection, our study introduces MultiBCD, a multimodal model that emulates the human diagnostic process for BC detection. Integrating an image classifier with GPT-4, it evaluates mammographic images alongside patient complaints. The model’s dual-head autoencoder efficiently processes image data, eliminating the need for manual lesion delineation, while GPT-4 extracts critical information from patient narratives.

MultiBCD demonstrates superior diagnostic accuracy and efficiency, achieving an F1 score of 80.15\% and a recall rate of 90.68\%, which marks an improvement over traditional methods. Furthermore, its design, emphasizing interpretability, aligns with the intuitive experience of medical consultations. The encouraging results of MultiBCD in BC detection indicate its potential for application in similar diagnostic contexts.

## Directory
- [Abstract](#abstract)
- [Dataset](#dataset)
- [Code Description](#code-description)
- [Installation](#installation)
- [Usage](#usage)
- [Citing & Authors](#citing--authors)
## Dataset
This study utilized the following two datasets:

- **The Chinese Mammography Database (CMMD):**  
  This database was conducted on 1,775 patients from China with benign or malignant breast disease who underwent mammography examination between July 2012 and January 2016. The database consists of 3,728 mammographies from these 1,775 patients, with biopsy-confirmed types of benign or malignant tumors. For 749 of these patients (1,498 mammographies), the database also includes patients' molecular subtypes. Image data were acquired on a GE Senographe DS mammography system. The data can be obtained from this [link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508)

- **Chinese Breast Disease Clinical Imaging Database(CBCID):**  
  This database includes 176 mammographic images and 84 corresponding patient complaints from 84 female breast disease patients.The data can be obtained from this [link](https://medbooks.ipmph.com/yx/imageLibrary/2578.html)

- **Mammography-Complaints Breast Database (MCBD):**
We present a multimodal dataset comprising 181 cases with mammographic images, complaints, and pathology results. To our knowledge, aside from the CBCID, no other dataset of this type is currently available. In support of advancing research in this area, we collected 181 samples from patients at The First Affiliated Hospital of Harbin Medical University, including both mammographic images and patient-reported complaints, along with corresponding pathology outcomes (117 benign, 64 malignant). 

## Code Description
- `cal_mean_std.py`: Calculate the mean and variance of mammography image datasets.
- `cal_para_quan.py`: Calculate the parameter quantity of models.
- `config.yaml`: A configuration file for setting up the project environment.
- `image_only.py`, `xgboost_muti.py`, `xgboost_text.py`: Decision-makers for text-only, image-only, and combined text and image analysis.
- `load_data.py`: Load datasets for processing.
- `losses.py`: Compute loss functions for model training.
- `model.py`: Build various types of models for image processing and analysis.
- `reconstruct.py`: Test the reconstruction effects of image autoencoder models.
- `test_classifier.py`, `test_vit.py`: Test various types of image classifiers.
- `train_ae.py`: Train autoencoder models.
- `train_classifier.py`, `train_vit.py`: Train various types of image classifiers.
- `utilss.py`: Contain general utility functions for the project.
- `requirement.txt`: The required python packages.
## Installation
```
pip install -r requirements.txt
```
## Usage
Specific parameters can be adjusted in config.yaml. The main experimental results can be reproduced through the following steps.  
Train the AutoEncoder
```
python train_ae.py
```
Train the Classifier
```
python train_classifier.py
```
Train the XGboost
```
python xgboost_muti.py
```

## Supplementary Information:
Due to copyright considerations, baseline methods with non-open-source code and CBCID are not included in this repository.

## Citing & Authors

This study has been accepted by _Biomedical Signal Processing and Control_ and is currently awaiting publication.

If you find this repository helpful, feel free to cite our publication -

MutiBCD: A Multimodal model that simulates the human diagnostic process for automated Breast Cancer Detection
