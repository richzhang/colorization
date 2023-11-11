from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, data_grayscale, data_color):

        self.data_grayscale = data_grayscale
        self.data_color = data_color

    def __len__(self):
        return len(self.data_grayscale)

    def __getitem__(self, idx):
        data_grayscale = self.data_grayscale[idx]
        data_color = self.data_color[idx]

        return data_grayscale, data_color