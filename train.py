import torch
from torch import optim

from model import DiT
from torch import nn


def train():
    model = DiT()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x = torch.rand(100, 1)
    y = torch.rand(100, 1)
    inputs = torch.cat((x, y), dim=1)
    targets = (x + y) * x

    # Train the model
    num_epochs = 10000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Test the model with an example
    test_input = torch.tensor([[.1, .7]])  # Example: x = 3, y = 4
    predicted = model(test_input).item()
    print(f"Test Input: x=.1, y=.7, Predicted: {predicted}, Actual: {(0.1 + 0.7) * 0.1}")

if __name__ == '__main__':
    train()
