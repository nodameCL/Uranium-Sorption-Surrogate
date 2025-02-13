from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import sys
import torch
import torch.utils.data
import os 
from ast import literal_eval

inputs = [r'$c_1$', r'$N_s$',r'$A_s$',r'$U^{4+}$',r'$Na^+$','pH', r'$logK_1$',
           r'$logK_2$',r'$logK_c$',r'$logK_a$', r'$logK_{UO_2^{2+}}$',r'$logK_{U^{4+}}$']

targets = [r'$\sigma_0$',r'$\sigma_{\beta}$',r'$\sigma_d$',r'$\psi_0$',r'$\psi_{\beta}$',r'$\psi_d$', r'$\equiv SO^-$',
           r'$\equiv SOH_2^+$', r'$\equiv SO^-:UO_2^{2+}$',r'$\equiv SO^-:Na^+$',r'$\equiv SOH^0$',r'$\equiv SO^-:U^{4+}$']

class SurfaceComplexationDataset(Dataset):
    """ surface complexation dataset """
    def __init__(self, 
                 root_dir, 
                 split = 'train'): 
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.root = root_dir 
        self.csv_file = os.path.join(root_dir, '{}_norm.csv'.format(split))
        self.data = pd.read_csv(self.csv_file)

        self.x = self.data[inputs]
        self.y = self.data[targets]

        self.x = torch.from_numpy(np.array(self.x).astype(np.float32))
        self.y = torch.from_numpy(np.array(self.y).astype(np.float32))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index): 
        return self.x[index], self.y[index]


if __name__ == '__main__':
    # d = SurfaceComplexationDataset(root_dir = '/home/chunhui/Documents/find_diff_coeffs/sigma_zeta_input', split='train')
    # d = SurfaceComplexationDataset_whole(root_dir = '/home/chunhui/Documents/find_diff_coeffs/train_val_test', split='test')
    # d = test_dataset(root_dir = '/home/chunhui/Documents/find_diff_coeffs/train_val_test', split='test')
    # data_dir = '/home/chunhui/Documents/SCM/exp/test_exp_on_3NN/fitted_LiCL'
    # d = SurfaceComplexationDataset(root_dir = '/home/chunhui/Documents/SCM/7points/points_811', split='val')
    data_dir = '/pscratch/sd/c/chunhui/Uranium_sorption_SCM_surrogate/dataset/'
    d = SurfaceComplexationDataset_csv(root_dir = data_dir, split='test')
    print("length of dataset is:", len(d), type(d))
   
    x, y = d[0] # get item when index = 0 
    print("shape of input: ", x.shape, type(x), x)
    print("shape of output: ", y.shape, type(y),  y)

    # data_dir = '/home/chunhui/Documents/SCM/7points/points_811'
    data_set = SurfaceComplexationDataset_csv(root_dir=data_dir, split='val')
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=2,
        shuffle=True,
        num_workers=0)
    
    print(len(data_loader), len(data_loader.dataset))
    for i, (inputs, targets) in enumerate(data_loader): 
        if i < 1: 
            print(inputs.shape, targets.shape, inputs[:, :3])
        else: 
            break
