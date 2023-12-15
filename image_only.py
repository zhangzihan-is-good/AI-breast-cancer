import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import Classifier
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class MutiDataset(Dataset):
    def __init__(self, xlsx_file, root_dir):
        """
        xlsx_file (string): Path to the xlsx file, contains ID and labels.
        root_dir (string): Directory containing all sub-folders.
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
        return final[0],final[1],label

# Instantiate MutiDataset
test_dataset = MutiDataset(xlsx_file='./Dataset_muti/testing_set.xlsx',
                      root_dir='./Dataset_muti/woman')
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

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

all_preds = []
all_labels = []
all_probs = []
# Testing loop
classifier.eval()  # Set model to evaluation mode
with torch.no_grad():
    for x1, x2, labels in tqdm(test_loader, desc='Testing', unit='batch'):
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        outputs = classifier(x1, x2)
        # Apply softmax function to get probability distribution
        probabilities = F.softmax(outputs, dim=1)
        print(probabilities)
        
        # Set probability threshold
        threshold = 0.73

        # Determine predicted class: if the maximum probability is less than the threshold, predict as 0, otherwise as the index of the maximum probability
        predicted = []
        for prob in probabilities:
            max_prob, idx = torch.max(prob, 0)
            if max_prob < threshold:
                predicted.append(0)
            else:
                predicted.append(idx.item())

        # Convert to Tensor
        predicted = torch.tensor(predicted)
        all_probs.extend(probabilities[:, 1].cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate P, R, F1 values
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
accuracy = accuracy_score(all_labels, all_preds)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}')
confu = confusion_matrix(all_labels, all_preds)
print(confu)

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('ROC')
print(f'ROC AUC: {roc_auc:.4f}')
