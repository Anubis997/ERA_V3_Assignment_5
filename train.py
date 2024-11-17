import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # All layers double channels, starting with 4
        self.features = nn.Sequential(
            # First conv block - start with 4 channels (1->4)
            nn.Conv2d(1, 4, kernel_size=3, padding=1),  # 28x28x4
            nn.BatchNorm2d(4),
            nn.ReLU(),
            
            # Second conv block - double channels (4->8)
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  # 28x28x8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14x8
            
            # Third conv block - double channels (8->16)
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 14x14x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Fourth conv block - maintain channels (16->16)
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 14x14x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7x16
            
            # Fifth conv block with stride=2 (16->16)
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # 4x4x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Sixth conv block (16->16)
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 4x4x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Dropout2d(0.25)
        )
        
        # Only one FC layer going directly to 10 classes
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced data transformations for better accuracy
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    
    # Decreased batch size from 128 to 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=4)
    
    # Initialize model
    model = MNIST_CNN().to(device)
    
    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel has {param_count} parameters")
    
    if param_count >= 25000:
        raise ValueError(f"Model has {param_count} parameters, which exceeds the limit of 25,000")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # Training
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    num_epochs = 1
    total_batches = len(trainloader)
    
    print(f'\nEpoch: 1/{num_epochs}')
    print('-' * 60)
    
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            current_accuracy = 100. * correct / total
            print(f'Batch: {batch_idx}/{total_batches} | Loss: {loss.item():.4f} | Accuracy: {current_accuracy:.2f}%')
    
    final_accuracy = 100. * correct / total
    print(f'\nFinal Training Accuracy: {final_accuracy:.2f}%')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'mnist_model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    
    return model, final_accuracy

if __name__ == "__main__":
    model, accuracy = train_model() 