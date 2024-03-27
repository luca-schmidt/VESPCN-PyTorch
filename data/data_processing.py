
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import preprocessing


'''

class MultiFrameDataset(Dataset):
    def __init__(self, data_path, component, scaling_factor, sequence_length, mean_value, std_value, start, end):
        super(MultiFrameDataset, self).__init__()
        #self.dataset = xr.open_dataset(data_path)[component].sel(time=slice(start, end))
        self.dataset = data_path[component]
        self.scaling_factor = scaling_factor
        self.sequence_length = sequence_length
        self.transforms = transforms.Compose([
           transforms.Normalize(mean=[mean_value], std=[std_value])])
        self.lr_data = downsampling(self.dataset, factor=self.scaling_factor).values

    def __len__(self):
        return len(self.dataset['time'].values) // self.sequence_length

    def __getitem__(self, idx):
        
        start_idx = idx * self.sequence_length

        hr_sequence = torch.stack([torch.from_numpy(self.dataset[start_idx + i, :, :32].values) for i in range(self.sequence_length)])
        lr_sequence = torch.stack([torch.from_numpy(self.lr_data[start_idx + i, :, :int(32/self.scaling_factor)]) for i in range(self.sequence_length)])

        hr_sequence = hr_sequence.unsqueeze(1)
        lr_sequence = lr_sequence.unsqueeze(1)

        hr_sequence[0] = self.transforms(hr_sequence[0])
        lr_sequence[0] = self.transforms(lr_sequence[0])

        return lr_sequence, hr_sequence

'''


class MultiFrameDataset(Dataset):
    def __init__(self, data_path, component, scaling_factor, sequence_length, mean_value, std_value, start, end):
        super().__init__()
        self.dataset = data_path[component] #xr.open_dataset(data_path)[component].sel(time=slice(start, end))
        self.scaling_factor = scaling_factor
        self.sequence_length = sequence_length
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=mean_value, std=std_value)])
        self.lr_data  = downsampling(self.dataset, factor=self.scaling_factor).values

    def __len__(self):
        return len(self.dataset['time'].values) - self.sequence_length + 1

    def __getitem__(self, idx):

        hr_sequence = torch.stack([torch.from_numpy(self.dataset[idx + i, :, :32].values) for i in range(self.sequence_length)])
        lr_sequence = torch.stack([torch.from_numpy(self.lr_data[idx + i, :, :int(32/self.scaling_factor)]) for i in range(self.sequence_length)])  

        hr_sequence = hr_sequence.unsqueeze(1)
        lr_sequence = lr_sequence.unsqueeze(1)

        lr_sequence = self.transforms(lr_sequence)
        hr_sequence = self.transforms(hr_sequence)

        return lr_sequence, hr_sequence
    

class MultiCompFrameDataset(Dataset):
    def __init__(self, data_path, component1, component2, scaling_factor, sequence_length, mean_value, std_value):
        super().__init__()
        self.data_path = data_path
        self.dataset1 = self.data_path[component1]
        self.dataset2 = self.data_path[component2]
        self.scaling_factor = scaling_factor
        self.sequence_length = sequence_length
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=mean_value, std=std_value)])
        self.lr_data1  = downsampling(self.dataset1, factor=self.scaling_factor).values
        self.lr_data2 = downsampling(self.dataset2, factor=self.scaling_factor).values


    def __len__(self):
        return len(self.dataset1['time'].values) - self.sequence_length + 1


    def __getitem__(self, idx):

        hr_sequence1 = torch.stack([torch.from_numpy(self.dataset1[idx + i, :, :32].values) for i in range(self.sequence_length)])
        lr_sequence1 = torch.stack([torch.from_numpy(self.lr_data1[idx + i, :, :int(32/self.scaling_factor)]) for i in range(self.sequence_length)])  

        hr_sequence2 = torch.stack([torch.from_numpy(self.dataset2[idx + i , :, :32].values) for i in range(self.sequence_length)])
        lr_sequence2 = torch.stack([torch.from_numpy(self.lr_data2[idx + i, :, :int(32/self.scaling_factor)]) for i in range(self.sequence_length)])


        hr_sequence1 = hr_sequence1.unsqueeze(1)
        lr_sequence1 = lr_sequence1.unsqueeze(1)
        hr_sequence2 = hr_sequence2.unsqueeze(1)
        lr_sequence2 = lr_sequence2.unsqueeze(1)

        lr_sequence = torch.cat([lr_sequence1, lr_sequence2], dim=1)
        hr_sequence = torch.cat([hr_sequence1, hr_sequence2], dim=1)

        lr_sequence = self.transforms(lr_sequence)
        hr_sequence = self.transforms(hr_sequence)

        return lr_sequence, hr_sequence
    

'''
class MultiCompFrameDataset(Dataset):
    def __init__(self, data_path, component1, component2, scaling_factor, sequence_length, mean_value, std_value):
        super().__init__()
        self.data_path = data_path
        self.dataset1 = self.data_path[component1]
        self.dataset2 = self.data_path[component2]
        self.scaling_factor = scaling_factor
        self.sequence_length = sequence_length
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=mean_value, std=std_value)])
        self.lr_data1  = downsampling(self.dataset1, factor=self.scaling_factor).values
        self.lr_data2 = downsampling(self.dataset2, factor=self.scaling_factor).values


    def __len__(self):
        print("Dataset len: ", len(self.dataset1['time'].values) - self.sequence_length + 1)
        return len(self.dataset1['time'].values) - self.sequence_length + 1


    def __getitem__(self, idx):

        start_idx = idx * self.sequence_length

        hr_sequence1 = torch.stack([torch.from_numpy(self.dataset1[start_idx + i, :, :32].values) for i in range(self.sequence_length)])
        lr_sequence1 = torch.stack([torch.from_numpy(self.lr_data1[start_idx + i, :, :int(32/self.scaling_factor)]) for i in range(self.sequence_length)])  

        hr_sequence2 = torch.stack([torch.from_numpy(self.dataset2[start_idx + i , :, :32].values) for i in range(self.sequence_length)])
        lr_sequence2 = torch.stack([torch.from_numpy(self.lr_data2[start_idx + i, :, :int(32/self.scaling_factor)]) for i in range(self.sequence_length)])


        hr_sequence1 = hr_sequence1.unsqueeze(1)
        lr_sequence1 = lr_sequence1.unsqueeze(1)
        hr_sequence2 = hr_sequence2.unsqueeze(1)
        lr_sequence2 = lr_sequence2.unsqueeze(1)

        lr_sequence = torch.cat([lr_sequence1, lr_sequence2], dim=1)
        hr_sequence = torch.cat([hr_sequence1, hr_sequence2], dim=1)

        lr_sequence = self.transforms(lr_sequence)
        hr_sequence = self.transforms(hr_sequence)

        return lr_sequence, hr_sequence
'''


def downsampling(xarray, factor):
  return xarray.isel(latitude=slice(0, None, factor), longitude=slice(0, None, factor))


def compute_stats(dataset, component, start, end):
  data_subset = dataset[component].sel(time=slice(start, end))
  mean_value = data_subset.mean().item()
  std_value = 1.0 #data_subset.std().item()
  return mean_value, std_value


def multi_stats(dataset, component_1, component_2, start, end):
  data_subset = dataset.sel(time=slice(start, end))
  mean_values = (data_subset[component_1].mean().values.item(), data_subset[component_2].mean().values.item())
  std_values = (1.0, 1.0) #(data_subset[component_1].std().values.item(), data_subset.std()[component_2].values.item())
  return mean_values, std_values


def rescale_data(data_path, component, start, end, scaler=None):
    ds = xr.open_dataset(data_path)
    data = ds[component].sel(time=slice(start, end))

    num, width, height = data.shape
    reshaped = data.values.reshape((num, width * height))

    # Use the provided scaler if it's not None, otherwise create a new one
    if scaler is None:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(reshaped)

    reshaped_norm = scaler.transform(reshaped)
    data_rescaled = reshaped_norm.reshape((num, width, height))

    data_rescaled_xr = xr.DataArray(data_rescaled, coords=data.coords, dims=data.dims, name=component)
    ds = xr.Dataset({component: data_rescaled_xr})

    return ds, scaler


def multi_rescale_data(data_path, component1, component2, start, end, scaler1=None, scaler2=None):
    ds = xr.open_dataset(data_path)
    data1 = ds[component1].sel(time=slice(start, end))
    data2 = ds[component2].sel(time=slice(start, end))

    num, width, height = data1.shape
    reshaped1 = data1.values.reshape((num, width * height))
    reshaped2 = data2.values.reshape((num, width * height))
    
    if scaler1 is None: 
       scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(reshaped1)
    if scaler2 is None: 
       scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(reshaped2)  

    data_rescaled1 = scaler1.transform(reshaped1).reshape((num, width, height))
    data_rescaled2 = scaler2.transform(reshaped2).reshape((num, width, height))

    data_rescaled_xr1 = xr.DataArray(data_rescaled1, coords=data1.coords, dims=data1.dims, name=component1)
    data_rescaled_xr2 = xr.DataArray(data_rescaled2, coords=data2.coords, dims=data2.dims, name=component2)

    ds = xr.Dataset({component1: data_rescaled_xr1, component2: data_rescaled_xr2})

    return ds, scaler1, scaler2


def global_rescale(data_path, start, end, custom_scale=None):
    ds = xr.open_dataset(data_path).sel(time=slice(start, end)).isel(longitude=slice(0, 32))
    returned_scale = {}

    for var in ds:
        if custom_scale and var in custom_scale:
            min_val = custom_scale[var]['min']
            max_val = custom_scale[var]['max']
        else:
            min_val = ds[var].min().values
            max_val = ds[var].max().values
            
            returned_scale[var] = {'min': min_val, 'max': max_val}

        ds[var] = (ds[var] - min_val) / (max_val - min_val)

    return ds, returned_scale



