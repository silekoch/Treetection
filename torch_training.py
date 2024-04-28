import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ForestCoverDataset import ForestCoverDataset
from models.unet.model import *
from helper import *
from torch import nn
from tqdm import tqdm

# Define training and validation phases
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, targets in tqdm(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f'Training Loss: {epoch_loss:.4f}')

def validate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f'Validation Loss: {epoch_loss:.4f}')

if __name__ == '__main__':
    train_dataset = ForestCoverDataset(mode='train', one_hot_masks=True)
    val_dataset = ForestCoverDataset(mode='val', one_hot_masks=True)

    BATCH_SIZE = 8

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")

    # Initialize the model and move it to the GPU if available
    model = UNet_tiny(num_classes=2).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10  # Set the number of epochs

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        validate(model, val_loader, criterion, device)
        print('-' * 10)
