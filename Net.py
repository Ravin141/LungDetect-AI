import torch
import torch.nn as nn

class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (1,160,160) -> (16,160,160)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # -> (16,80,80)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> (32,80,80)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # -> (32,40,40)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # -> (64,40,40)
            nn.ReLU(),
            nn.MaxPool2d(2)                                        # -> (64,20,20)
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 128),  # 64 * 20 * 20 = 25,600
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
