import torch


class CustomDataset(torch.utils.data.Dataset):
    """Refer to https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {
            'features': self.X[idx],
            'labels': torch.tensor(self.y[idx]),
        }
        return item


def create_dataset(X, y):
    return CustomDataset(X, y)
