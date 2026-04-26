import torch
import torch.nn as nn

class StandardCNN(nn.Module):
    def __init__(self, in_channels=1, input_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        
        # Calculate flat features:
        # Conv1: (input_size - 5 + 1)
        # Conv2: (Conv1_out - 5 + 1)
        # Pool: Conv2_out / 2
        conv_out_size = (input_size - 4 - 4) // 2
        self.fc = nn.Linear(32 * conv_out_size * conv_out_size, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
