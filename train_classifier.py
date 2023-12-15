import torch
from model import Classifier, VGG, ResNet50, GoogleNet, AlexNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from load_data import class_Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import utilss
from transformers import ViTForImageClassification

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x1, x2, labels in train_loader:
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x1, x2, labels in val_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss
# Set random seed
utilss.randomseed(0)

dataset = class_Dataset(root_dir='./Data', label_path='./Data_Label.xlsx')

# Calculate split points
total_size = len(dataset)
print(total_size)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Sequentially split the dataset
train_dataset = Subset(dataset, range(0, train_size))
val_dataset = Subset(dataset, range(train_size, train_size + val_size))

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

config = utilss.load_config('./config.yaml')
num_epochs = config.get('class_num_epochs')
learning_rate = config.get('classify_lr')
model_name = config.get('model_name')
print(model_name)
if model_name in ["all"]:
    
    classifier = Classifier()
    # Path to the saved model file
    model_path = './best_ae_model_{}.pth'.format(model_name)
    # Load model state
    state_dict = torch.load(model_path)
    print("{} has been loaded".format(model_name))
    # Apply state to the model
    classifier.load_partial_state_dict(state_dict, load_encoder=True, load_decoder=False)
    optimizer = torch.optim.Adam([
    {'params': classifier.encoder.parameters(), 'lr': 5e-4},
    {'params': classifier.classify.parameters(), 'lr': learning_rate}
    ])

elif model_name == "nop":
    classifier = Classifier()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

elif model_name == "VGG":
    classifier = VGG()
    print("VGG has been loaded")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

elif model_name == "ResNet50":
    classifier = ResNet50()
    print("ResNet50 has been loaded")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

elif model_name == "GoogleNet":
    classifier = GoogleNet()
    print("GoogleNet has been loaded")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

elif model_name == "AlexNet":
    classifier = AlexNet()
    print("AlexNet has been loaded")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

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
    
# Training process
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_path = 'best_classifier_model_{}.pth'.format(model_name)
for epoch in tqdm(range(num_epochs), desc='Epochs'):
    train_loss = train_epoch(classifier, train_loader, optimizer, criterion, device)
    val_loss = validate(classifier, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(classifier.state_dict(), best_model_path)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_plot_{}.png'.format(model_name))
