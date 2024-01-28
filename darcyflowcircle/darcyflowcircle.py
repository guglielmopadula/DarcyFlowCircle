import numpy as np
import torch
import os
from torch.utils.data import TensorDataset
import scipy.sparse



class DarcyFlowCircle():
    def __init__(self, batch_size):
        from numpy.random import Generator, PCG64
        self.rg = Generator(PCG64(42))
        test_indices=self.rg.choice(600,size=100, replace=False)
        train_indices=np.setdiff1d(np.arange(600),test_indices)
        self.batch_size = batch_size
        self.data_directory = os.path.join(os.path.dirname(__file__), 'data')
        self.points=torch.tensor(np.load(os.path.join(self.data_directory, 'points.npy')),dtype=torch.float32)
        self.sc_matrix=scipy.sparse.load_npz(os.path.join(self.data_directory, 'sc_matrix.npz'))
        V=torch.tensor(np.load(os.path.join(self.data_directory, 'v.npy')),dtype=torch.float32)
        U=torch.tensor(np.load(os.path.join(self.data_directory, 'u.npy')),dtype=torch.float32)
        self.V_train=V[train_indices]
        self.V_test=V[test_indices]
        del V
        self.U_train=U[train_indices]
        self.U_test=U[test_indices]
        del U
        self.train_dataset=TensorDataset(self.V_train,self.U_train)
        self.test_dataset=TensorDataset(self.V_test,self.U_test)
        self.train_loader=torch.utils.data.DataLoader(self.train_dataset,batch_size=batch_size,shuffle=False)
        self.test_loader=torch.utils.data.DataLoader(self.test_dataset,batch_size=batch_size,shuffle=False)
        self.pod_basis=torch.tensor(np.load(os.path.join(self.data_directory, 'pod_basis.npy')),dtype=torch.float32)
        self.pod_coeff=torch.tensor(np.load(os.path.join(self.data_directory, 'pod_coeff.npy')),dtype=torch.float32)
