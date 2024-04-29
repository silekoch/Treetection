import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ForestCoverDataset import ForestCoverDataset
from models.unet.model import *
from helper import *
from torch import nn
from tqdm import tqdm
import torchmetrics

# Define training and validation phases
def train_one_epoch(model, dataloader, optimizer, criterion, device, metrics):
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
        for metric in metrics.values():
            metric(outputs, targets)

    epoch_loss = running_loss / len(dataloader)
    for name, metric in metrics.items():
        print(f'{name}: {metric.compute()}')
        metric.reset()
    print(f'Training Loss: {epoch_loss:.4f}')

def validate(model, dataloader, criterion, device, metrics):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            for metric in metrics.values():
                metric(outputs, targets)

    epoch_loss = running_loss / len(dataloader)
    for name, metric in metrics.items():
        print(f'{name}: {metric.compute()}')
        metric.reset()
    print(f'Validation Loss: {epoch_loss:.4f}')

if __name__ == '__main__':
    train_dataset = ForestCoverDataset(mode='train', one_hot_masks=True)
    val_dataset = ForestCoverDataset(mode='val', one_hot_masks=True)

    BATCH_SIZE = 16

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
    num_epochs = 6  # Set the number of epochs

    metric_iou = torchmetrics.classification.BinaryJaccardIndex().to(device)
    metric_f1 = torchmetrics.classification.BinaryF1Score().to(device)
    metric_acc = torchmetrics.classification.BinaryAccuracy().to(device)

    metrics = {
        'iou': metric_iou,
        'f1': metric_f1,
        'acc': metric_acc
    }
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')        
        train_one_epoch(model, train_loader, optimizer, criterion, device, metrics)
        validate(model, val_loader, criterion, device, metrics)
        torch.save(model.state_dict(), f'model_{epoch}.pth')
        print('-' * 10)

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
