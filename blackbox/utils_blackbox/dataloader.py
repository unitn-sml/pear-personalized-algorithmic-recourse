import numpy as np

import torch
from torch.utils.data import Dataset

def fix_adult(data):
    # https://www.kaggle.com/code/alokevil/simple-eda-for-beginners
    data.dropna(inplace=True)
    attrib, counts = np.unique(data['workclass'], return_counts = True)
    most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
    data['workclass'] = data['workclass'].apply(lambda x: most_freq_attrib if x=='?' else x)

    attrib, counts = np.unique(data['occupation'], return_counts = True)
    most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
    data['occupation'] = data['occupation'].apply(lambda x: most_freq_attrib if x=='?' else x)

    attrib, counts = np.unique(data['native_country'], return_counts = True)
    most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
    data['native_country'] = data['native_country'].apply(lambda x: most_freq_attrib if x=='?' else x)

    return data

class Data(Dataset):

    def __init__(self, X, y):
        self.data = X
        self.y = y

    def feature_size(self):
        return len(self.data.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx].values).float()
        response = torch.tensor(self.y.iloc[idx]).float()
        return features, response