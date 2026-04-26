import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []

        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            label = int(folder.split("_")[0])

            for file in os.listdir(folder_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(folder_path, file), label)
                    )

    def __len__(self):
        return len(self.samples)

    def augment(self, img):
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)

        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        img = self.augment(img)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return img, label