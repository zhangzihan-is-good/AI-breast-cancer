import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from load_data import class_Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import utilss
from transformers import ViTForImageClassification
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc

# Set random seed
utilss.randomseed(0)

dataset = class_Dataset(root_dir='./Data', label_path='./Data_Label.xlsx')

# Calculate split points
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Sequentially split the dataset
test_dataset = Subset(dataset, range(train_size + val_size, total_size))

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

config = utilss.load_config('./config.yaml')
num_epochs = config.get('class_num_epochs')
model_name = config.get('test_model')
print(model_name)
best_model_path = 'best_classifier_model_VIT.pth'

classifier = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)
criterion = nn.CrossEntropyLoss()
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
print("ViT has been loaded")

def test(model, test_loader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for x1, x2, labels in test_loader:
            x = torch.cat([x1, x2, x1], dim=1)
            x, labels = x.to(device), labels.to(device)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            outputs = model(x)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predictions = (probabilities[:, 1] >= threshold).int()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)

    return precision, recall, f1, conf_matrix, fpr, tpr, roc_auc

# Test the model
precision, recall, f1, conf_matrix, fpr, tpr, roc_auc = test(classifier, test_loader, device)

# Print metrics
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, accurcy: {roc_auc:.4f}')
print(conf_matrix)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROCVIT')
 



