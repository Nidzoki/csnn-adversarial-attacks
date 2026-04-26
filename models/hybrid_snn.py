import torch
import snntorch as snn
import torch.nn as nn
import torch.nn.functional as F

class HybridCSNN(nn.Module):
    def __init__(self, in_channels=1, input_size=28, num_steps=25):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2)

        conv_out_size = (input_size - 4 - 4) // 2

        # LIF spike layer
        self.lif = snn.Leaky(beta=0.9, spike_grad=snn.surrogate.fast_sigmoid(), init_hidden=False)

        # Fully Connected output
        self.fc = nn.Linear(32 * conv_out_size * conv_out_size, 10)

    def forward(self, x):
        # 1. Inicijalizacija stanja
        mem = self.lif.init_leaky()

        # 2. CNN Feature extraction
        cur = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(x)))))
        cur = cur.view(cur.size(0), -1)

        spk_rec = []

        # 3. Vremenska petlja
        for step in range(self.num_steps):
            spk, mem = self.lif(cur, mem) 
            out = self.fc(spk)
            spk_rec.append(out)

        return torch.stack(spk_rec, dim=0)