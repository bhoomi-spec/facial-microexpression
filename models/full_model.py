import torch
import torch.nn as nn
from models.cnn import CNN


class FullModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.cnn = CNN()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            feat = self.cnn(dummy)
            feat_dim = feat.shape[1]

        print("Detected feature size:", feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.cnn(x)
        out = self.classifier(feat)
        return out