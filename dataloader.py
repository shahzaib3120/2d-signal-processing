from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('data/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

class BitmapDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.noisy_bitmaps = np.array(self.dataset['bitmaps']['noisy'])
        self.original_bitmaps = np.array(self.dataset['bitmaps']['original'])
        self.n_noisy = len(self.noisy_bitmaps)
        self.n_original = len(self.original_bitmaps)
        self.label2id = {
            0: 'Original',
            1: 'Noisy'
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def __len__(self):
        return self.n_noisy + self.n_original

    def __getitem__(self, idx):
        return {
            "bitmap": self.noisy_bitmaps[idx] if idx < self.n_noisy else self.original_bitmaps[idx - self.n_noisy],
            "label": 1 if idx < self.n_noisy else 0
        }

dataset = BitmapDataset(dataset)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    bitmaps = batch['bitmap']
    labels = batch['label']
    for i in range(len(bitmaps)):
        plt.imshow(bitmaps[i], cmap='gray', aspect='auto')
        plt.title(dataset.label2id[labels[i]])
        plt.show()
    break