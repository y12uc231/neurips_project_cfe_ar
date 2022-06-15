import torch
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class DataLoader_Tabular(data.Dataset):
    def __init__(self, path, filename, label, scale='minmax'):
        
        """
        Load training dataset
        :param path: string with path to training set
        :param label: string, column name for label
        :param scale: string; either 'minmax' or 'standard'
        :return: tensor with training data
        """
        
        # Load dataset
        self.dataset = pd.read_csv(path + filename)
        self.target = label
        
        # Cleaning Routine

        # Save target and predictors
        self.X = self.dataset.drop(self.target, axis=1)
        
        # Save feature names
        self.feature_names = self.X.columns.to_list()
        self.target_name = label

        # Transform data
        if scale == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scale == 'standard':
            self.scaler = StandardScaler()
            
        self.scaler.fit_transform(self.X)
        
        self.data = self.scaler.transform(self.X)
        self.targets = self.dataset[self.target]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # select correct row with idx
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return self.data[idx], self.targets.values[idx], idx

    def get_number_of_features(self):
        return self.data.shape[1]
    
    def get_number_of_instances(self):
        return self.data.shape[0]
    

def return_loaders(data_name, is_tabular, batch_size=32, transform=None, scaler='minmax'):
    
    if is_tabular:
        transform = None
    else:
        if transform is not None:
            transform = transform
        else:
            # Standard Transforms
            if data_name == 'mnist':
                transform = transforms.Compose([transforms.ToTensor()
                                                ])
            # Not supported data sets
            else:
                raise ValueError
            
    # Dictionary
    dict = {'mnist': ('MNIST', transform, is_tabular, None),
            'admission': ('Admission', transform, is_tabular, 'zfya'),
            'adult': ('Adult', transform, is_tabular, 'income'),
            'compas': ('COMPAS', transform, is_tabular, 'risk'),
            'twomoons': ('TwoMoons', transform, is_tabular, 'label'),
            }
    
    if dict[data_name][2]:
        file_train = data_name + '-train.csv'
        file_test = data_name + '-test.csv'
    
        dataset_train = DataLoader_Tabular(path='./Data_Sets/' + dict[data_name][0] + '/',
                                           filename=file_train, label=dict[data_name][3], scale=scaler)
    
        dataset_test = DataLoader_Tabular(path='./Data_Sets/' + dict[data_name][0] + '/',
                                          filename=file_test, label=dict[data_name][3], scale=scaler)
    else:
        dataset_train = MNIST(root='./Data_Sets/' + dict[data_name][0] + '/', train=True, download=True,
                              transform=dict[data_name][1])
        dataset_test = MNIST(root='./Data_Sets/' + dict[data_name][0] + '/', train=False, download=True,
                             transform=dict[data_name][1])


    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    
    return trainloader, testloader

