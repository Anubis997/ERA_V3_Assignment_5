import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from train import MNIST_CNN
import glob
import pytest
import torch.optim as optim

def get_latest_model():
    model_files = glob.glob('mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model file found")
    latest_model = max(model_files)
    return latest_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNIST_CNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_input_output():
    model = MNIST_CNN()
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    # Load the latest model
    model = MNIST_CNN()
    latest_model = get_latest_model()
    model.load_state_dict(torch.load(latest_model, map_location=torch.device('cpu')))
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                           shuffle=False, num_workers=2)
    
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data, target in testloader:
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be > 95%"

def test_model_gradient_flow():
    """Test if gradients are flowing properly through the model"""
    model = MNIST_CNN()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy data
    dummy_input = torch.randn(1, 1, 28, 28)
    dummy_target = torch.tensor([5])  # Random target class
    
    # Forward pass
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    
    # Check if gradients exist and are not zero
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            has_gradients = True
            break
    
    assert has_gradients, "Model has no meaningful gradients"

def test_model_overfitting_single_batch():
    """Test if model can overfit to a single batch (sanity check)"""
    model = MNIST_CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create small dummy dataset
    x = torch.randn(5, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4])
    
    # Train for few iterations
    for _ in range(50):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Check if model can predict the training data
    with torch.no_grad():
        output = model(x)
        _, predicted = output.max(1)
        accuracy = (predicted == y).float().mean().item()
    
    assert accuracy > 0.8, "Model unable to overfit single batch"

def test_model_robustness():
    """Test model's robustness to input noise"""
    model = MNIST_CNN()
    latest_model = get_latest_model()
    model.load_state_dict(torch.load(latest_model, map_location=torch.device('cpu')))
    model.eval()
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    
    # Test original and noisy versions
    x, y = testset[0]
    x = x.unsqueeze(0)
    
    # Add noise
    noise = torch.randn_like(x) * 0.1
    x_noisy = x + noise
    
    with torch.no_grad():
        out_original = model(x)
        out_noisy = model(x_noisy)
        
        # Get predictions
        _, pred_original = out_original.max(1)
        _, pred_noisy = out_noisy.max(1)
    
    # Check if predictions match
    assert pred_original == pred_noisy, "Model predictions change significantly with small noise"

if __name__ == "__main__":
    pytest.main([__file__]) 