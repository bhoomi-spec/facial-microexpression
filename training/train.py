import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.full_model import FullModel
from utils.dataset import SequenceDataset

# -------- DATASET --------
dataset = SequenceDataset("data/sequences")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -------- MODEL --------
model = FullModel()

# 🔥 CLASS WEIGHTS (important)
weights = torch.tensor([1.0, 2.0, 2.0])  # adjust if needed
loss_fn = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------- TRAIN --------
for epoch in range(25):   # 🔥 increased epochs
    total_loss = 0

    for seq, label in loader:
        optimizer.zero_grad()

        output = model(seq)
        loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -------- SAVE --------
torch.save(model.state_dict(), "model.pth")
print("Model saved")