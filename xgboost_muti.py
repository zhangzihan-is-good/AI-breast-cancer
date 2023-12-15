import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import Classifier
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class MutiDataset(Dataset):
    def __init__(self, xlsx_file, root_dir):
        """
        xlsx_file (string): Path to the xlsx file, contains ID and labels.
        root_dir (string): Directory containing all subfolders.
        """
        self.data_frame = pd.read_excel(xlsx_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.data_frame.iloc[idx, 0]  # Read ID
        img_id = '{:03d}'.format(img_id)
        label = self.data_frame.iloc[idx, 5]  # Read the sixth column as label
        label = 1 if label == 'Malignant' else 0  # Convert label to numeric

        # Construct the path of the image
        img_folder = os.path.join(self.root_dir, img_id)
        img_names = os.listdir(img_folder)
        img_paths = [os.path.join(img_folder, name) for name in img_names]

        # Load images
        images = [Image.open(img_path) for img_path in img_paths]
        final = []
        for ima in images:
            # Contrast enhancement - using CLAHE
            np_image = np.array(ima)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(np_image)

            # Convert to PyTorch tensor
            image_tensor = torch.from_numpy(enhanced_image).unsqueeze(0).float()

            # Normalize
            normalize = transforms.Normalize(mean=[22.05], std=[42.99])
            norm_image = normalize(image_tensor)
            final.append(norm_image)
        
        # Return images and label
        return final[0], final[1], label

best_model_path = './best_classifier_model_all.pth'
classifier = Classifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    # First move the model to GPU
    classifier = classifier.to('cuda')
    # Wrap the model with DataParallel
    classifier = nn.DataParallel(classifier)
else:
    # If only one GPU or only CPU, directly move the model to the device
    classifier = classifier.to(device)
# After training is complete, load the best model
classifier.load_state_dict(torch.load(best_model_path))

# Instantiate MutiDataset
train_dataset = MutiDataset(xlsx_file='./Dataset_muti/training_set.xlsx', root_dir='./Dataset_muti/woman')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)
# Array to store malignant probabilities
malignant_probs_train = []
# Testing loop
classifier.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for x1, x2, labels in tqdm(train_loader, desc='Testing', unit='batch'):
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        outputs = classifier(x1, x2)
        # Apply softmax function to get probability distribution
        probabilities = F.softmax(outputs, dim=1)
        # Extract and store malignant probabilities
        malignant_probs_train.extend(probabilities[:, 1].cpu().numpy())  # Extract malignant probability and add to array

# Instantiate MutiDataset
test_dataset = MutiDataset(xlsx_file='./Dataset_muti/testing_set.xlsx', root_dir='./Dataset_muti/woman')
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
# Array to store malignant probabilities
malignant_probs_test = []
# Testing loop
classifier.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for x1, x2, labels in tqdm(test_loader, desc='Testing', unit='batch'):
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        outputs = classifier(x1, x2)
        # Apply softmax function to get probability distribution
        probabilities = F.softmax(outputs, dim=1)
        # Extract and store malignant probabilities
        malignant_probs_test.extend(probabilities[:, 1].cpu().numpy())  # Extract malignant probability and add to array

# Load training and testing data
train_file_path = './Dataset_muti/training_set.xlsx'
test_file_path = './Dataset_muti/testing_set.xlsx'
train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)

# Add malignant probability feature to training and testing data
train_data['Malignant_Probability'] = malignant_probs_train
test_data['Malignant_Probability'] = malignant_probs_test
# Apply logarithmic transformation to 'Malignant_Probability'
train_data['Malignant_Probability'] = np.log1p(train_data['Malignant_Probability'])
test_data['Malignant_Probability'] = np.log1p(test_data['Malignant_Probability'])

# Initialize label encoder
label_encoder = LabelEncoder()

for column in ['Pain', 'Texture', 'Skin Changes', 'Nipple Discharge']:
    train_data[column] = label_encoder.fit_transform(train_data[column])
    test_data[column] = label_encoder.transform(test_data[column])

# Extract features and labels from training data
X_train = train_data[['Pain', 'Texture', 'Skin Changes', 'Nipple Discharge', 'Malignant_Probability']]
y_train = train_data['Benign or Malignant'].map({'Malignant': 1, 'Benign': 0})

# Extract features and labels from testing data
X_test = test_data[['Pain', 'Texture', 'Skin Changes', 'Nipple Discharge', 'Malignant_Probability']]
y_test = test_data['Benign or Malignant'].map({'Malignant': 1, 'Benign': 0})

# Initialize XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth = 4, learning_rate = 0.01, n_estimators = 200, subsample = 0.8, colsample_bytree = 0.2, gamma = 0)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict using the model
y_pred = xgb_model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print performance metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')
