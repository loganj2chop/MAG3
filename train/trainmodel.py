import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Ensure the image has the correct shape and data type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Ensure the image is in the correct shape (256, 256, 3)
        if image.shape[0] == 1:  # Handle single-channel images
            image = image.squeeze(0)
        
        image = Image.fromarray(image)

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
        

X_train = np.load('original_new_1_train.npy')
X_train = np.stack([X_train] * 3, axis=-1)

X_test = np.load('original_new_1_test.npy')
X_test = np.stack([X_test] * 3, axis=-1)

y_train_df = pd.read_csv('ids_with_metrics_fin_mix1_train.csv')
y_train = y_train_df['Bad'].values
y_train = np.clip(y_train, 0, 1)

y_test_df = pd.read_csv('ids_with_metrics_fin_mix1_test.csv')
y_test = y_test_df['Bad'].values
y_test = np.clip(y_test, 0, 1)

# Define transformation for PyTorch
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create training and test datasets
train_dataset = ImageDataset(X_train, y_train, transform=transform)
test_dataset = ImageDataset(X_test, y_test, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 1)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50 = resnet50.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)

# Training loop with early learning rate adjustment and checkpointing
def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    model.train()
    best_loss = float('inf')
    patience_counter = 0  # To track the number of epochs with no improvement

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Check for improvement and save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), "best_original_model_1.pth")  # Save best model
            print(f"Best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
        else:
            patience_counter += 1

        # If the loss has not improved for 5 consecutive epochs, adjust the learning rate
        if patience_counter >= 7:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5  # Reduce the learning rate by half
            print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']}")
            patience_counter = 0  # Reset patience counter after learning rate adjustment

    # Save the final model after training
    torch.save(model.state_dict(), "original_final_model_1.pth")
    print("Final model saved after training.")

# Train the model
train_model(resnet50, train_loader, criterion, optimizer, num_epochs=200)

# Define criterion for per-sample loss in evaluation
criterion_per_sample = nn.BCEWithLogitsLoss(reduction='none')

# Evaluation function to calculate probabilities, loss, and append to y_test_df
def evaluate_and_save(model, dataloader, y_test_df):
    model.eval()
    probabilities = []
    losses = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()

            # Calculate per-sample loss
            batch_losses = criterion_per_sample(outputs, labels).cpu().numpy().squeeze()

            # Append probabilities and losses
            probabilities.extend(probs)
            losses.extend(batch_losses if isinstance(batch_losses, list) else batch_losses.tolist())

    # Add probabilities and loss columns to y_test_df
    y_test_df['probability'] = probabilities
    y_test_df['loss'] = losses

    # Save the updated DataFrame
    y_test_df.to_csv('orignal_ids_with_metrics_fin_mix1_test_with_predictions.csv', index=False)
    print("Saved predictions and losses to ids_with_metrics_fin_mix1_test_with_predictions.csv")

# Call the evaluation and save function
evaluate_and_save(resnet50, test_loader, y_test_df)