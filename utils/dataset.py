import os
import cv2
import torch
import numpy as np
import re
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len=5):
        self.data_dir = data_dir
        self.samples = os.listdir(data_dir)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_path = os.path.join(self.data_dir, sample_name)

        images = sorted(os.listdir(sample_path))

        seq = []

        for img_name in images[:self.seq_len]:
            img_path = os.path.join(sample_path, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0

            # convert HWC → CHW
            img = np.transpose(img, (2, 0, 1))

            seq.append(img)

        # 🔥 convert list → numpy → tensor (fix warning)
        seq = torch.tensor(np.array(seq), dtype=torch.float32)

        # 🔥 FIXED LABEL EXTRACTION (works for 0.1seq1 etc.)
        numbers = re.findall(r'\d+', sample_name)

        if len(numbers) == 0:
            raise ValueError(f"Invalid folder name: {sample_name}")

        label = int(numbers[0][0])  # take first digit

        label = torch.tensor(label, dtype=torch.long)

        return seq, label