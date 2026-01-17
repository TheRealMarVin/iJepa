from torch.utils.data import Dataset


class IJEPADatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]

        return image
