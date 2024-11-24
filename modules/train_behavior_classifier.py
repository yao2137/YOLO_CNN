import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def train_behavior_classifier(data_dir, model_save_path, num_classes=5, epochs=10, batch_size=32, lr=0.001):
    """
    Fine-tunes a ResNet model on the specified dataset.

    Args:
        data_dir (str): Path to the dataset directory with 'train' and 'val' subdirectories.
        model_save_path (str): Path to save the fine-tuned model.
        num_classes (int): Number of behavior categories.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
    """
    # Define data transformations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Update the fully connected layer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # Train phase
        model.train()
        train_loss = 0
        correct, total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validate phase
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Example usage
    data_dir = "data/kinetics"
    model_save_path = "models/behavior/behavior_classifier.pth"
    train_behavior_classifier(data_dir, model_save_path, num_classes=5, epochs=10, batch_size=32, lr=0.001)