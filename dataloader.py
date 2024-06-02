from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np

class BitmapDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.noisy_bitmaps = np.array(self.dataset['bitmaps']['noisy'])
        self.original_bitmaps = np.array(self.dataset['bitmaps']['original'])
        self.n_noisy = len(self.noisy_bitmaps)
        self.n_original = len(self.original_bitmaps)
        self.id2label = {
            0: 'Original',
            1: 'Noisy'
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        print(f'Noisy: {self.n_noisy}, Original: {self.n_original}')
        
    def __len__(self):
        return self.n_noisy + self.n_original

    def __getitem__(self, idx):
        return {
            "bitmap": self.noisy_bitmaps[idx] if idx < self.n_noisy else self.original_bitmaps[idx - self.n_noisy],
            "label": 1 if idx < self.n_noisy else 0
        }

if __name__ == '__main__':

    with open('data/dataset.pkl', 'rb') as f:
        pk_dataset = pickle.load(f)
    dataset = BitmapDataset(pk_dataset)
    del pk_dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # print data shape
    for batch in dataloader:
        bitmaps = batch['bitmap']
        labels = batch['label']
        print(bitmaps.shape)
        print(labels)
        for i in range(len(bitmaps)):
            plt.imshow(bitmaps[i], cmap='gray', aspect='auto')
            plt.title(dataset.id2label[labels[i].item()])
            plt.show()
        break