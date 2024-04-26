import torch
import torch.nn as nn
import torch.optim as optim

# Data for XOR
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

# Define the model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # First layer: 2 input features (XOR inputs), 2 hidden nodes
        self.layer1 = nn.Linear(2, 2)
        # Activation function between layers
        self.activation = nn.Sigmoid()
        # Output layer: 2 inputs from hidden layer, 1 output
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

# Instantiate the model
model = XORModel()

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Testing the model
with torch.no_grad():
    predicted = model(X)
    predicted = torch.round(torch.sigmoid(predicted))
    print(f'Predicted:\n{predicted}\nActual:\n{y}')
