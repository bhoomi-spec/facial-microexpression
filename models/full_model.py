import torch
import torch.nn as nn
from models.cnn import CNN

class FullModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.cnn = CNN()

        # 🔥 AUTO-DETECT FEATURE SIZE (no guessing)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            feat = self.cnn(dummy)
            self.feature_dim = feat.shape[1]

        print("Detected feature size:", self.feature_dim)

        # 🔥 CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq):
        # x_seq: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x_seq.shape

        features = []

        for t in range(seq_len):
            frame = x_seq[:, t]          # (batch, C, H, W)
            f = self.cnn(frame)          # (batch, feature_dim)
            features.append(f)

        # 🔥 temporal average
        features = torch.stack(features, dim=1)   # (batch, seq_len, feat)
        features = torch.mean(features, dim=1)    # (batch, feat)

        output = self.classifier(features)

        return output