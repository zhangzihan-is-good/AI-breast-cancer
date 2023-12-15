import torch
from model import Classifier,VGG,ResNet50,AlexNet,GoogleNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from load_data import class_Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import utilss
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

# Set random seed
utilss.randomseed(0)

dataset = class_Dataset(root_dir='./Data', label_path='./Data_Label.xlsx')

# Calculate split points
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Split the dataset in order
test_dataset = Subset(dataset, range(train_size + val_size, total_size))

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

config = utilss.load_config('./config.yaml')
model_name = config.get('test_model')
print("test {}".format(model_name))
best_model_path = './best_classifier_model_{}.pth'.format(model_name)

if model_name in ["all","nop"]:
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

else:  
    if model_name == "VGG":
        classifier = VGG()
        
    if model_name == "AlexNet":
        classifier = AlexNet()
    
    if model_name == "ResNet50":
        classifier = ResNet50()
        
    if model_name == "GoogleNet":
        classifier = GoogleNet()
    
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
    classifier.load_state_dict(torch.load(best_model_path))
    
all_preds = []
all_labels = []
all_probs = []
# Testing loop
classifier.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for x1, x2, labels in tqdm(test_loader, desc='Testing', unit='batch'):
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        outputs = classifier(x1, x2)
        # Apply softmax function to get probability distribution
        probabilities = F.softmax(outputs, dim=1)
        print(probabilities)
        
        # Set probability threshold
        threshold = 0.64

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
plt.title('ROC'.format(model_name))
plt.legend(loc="lower right")
plt.savefig('ROC{}'.format(model_name))
print(f'ROC AUC: {roc_auc:.4f}')