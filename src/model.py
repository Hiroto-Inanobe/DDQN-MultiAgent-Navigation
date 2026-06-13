import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, in_ch, obs_size, out_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * obs_size * obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
