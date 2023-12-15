import torch
from torch.utils.data import DataLoader, random_split
from model import AE
from losses import ae_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from load_data import PretrainDataset
import utilss

# Set random seed
utilss.randomseed(42)

config = utilss.load_config('./config.yaml')
num_epochs = config.get('num_epochs')
learning_rate = config.get('pretrain_lr')

# Load the dataset and split into training and validation sets
dataset = PretrainDataset(root_dir='./Data')
print(len(dataset))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ae = AE()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    ae = ae.to('cuda')
    ae = nn.DataParallel(ae)
else:
    ae = ae.to(device)

optimizer = torch.optim.Adam(ae.parameters(), lr = learning_rate)
epoch_losses = []  # Store average loss for each epoch
val_losses = []  # Store validation loss for each epoch
best_loss = float('inf')
best_epoch = 0
best_model_path = ('best_ae_model.pth')
best_models = {}

for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
    ae.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = ae(data)
        loss = ae_loss(recon_batch, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    average_train_loss = train_loss / len(train_loader)
    epoch_losses.append(average_train_loss)

    # Validation step
    ae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            recon_batch = ae(data)
            loss = ae_loss(recon_batch, data)
            val_loss += loss.item()
    average_val_loss = val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    # Update best model information
    if average_val_loss < best_loss:
        best_loss = average_val_loss
        best_epoch = epoch
        torch.save(ae.state_dict(), best_model_path)

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

print(f'Best model was saved at epoch {best_epoch + 1} with validation loss {best_loss:.4f}')

# Plotting the loss graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='blue', label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, marker='x', linestyle='-', color='red', label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_plot.png')
