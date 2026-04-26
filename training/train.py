import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.full_model import FullModel
from utils.dataset import EmotionDataset

dataset = EmotionDataset("data/sequences")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = FullModel()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(25):
    total_loss = 0
    correct = 0
    total = 0

    for img, label in loader:
        optimizer.zero_grad()

        output = model(img)
        loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = torch.argmax(output, dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    acc = 100 * correct / total

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "model.pth")
print("Model saved")