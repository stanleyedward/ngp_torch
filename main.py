import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

class NGP(nn.Module):
    def __init__(self, T, Nl, L, device, aabb_scale, F=2):
        super(NGP, self).__init__()
        # refer page 4 of paper
        
        self.T = T # max entries per level(hash table size)
        self.Nl = Nl # coarse / fine resolution
        self.F = F # number of feature deimensions per entry
        self.L = L # no of levels / encoding directions 
        self.aabb_scale = aabb_scale # scale of the scene
        
        self.lookup_tables = torch.nn.ParameterDict(
            {str(i): torch.nn.Parameter((torch.rand(
                (T,F), device=device) * 2-1) * 1e-4) for i in range(len(Nl))
            })
        
        # 3 prime nos used for hashing position xyz to map it into an index in the table
        self.pi1 = 1
        self.pi2 = 2_654_435_761
        self.pi3 = 805_459_861
        
        self.density = nn.Sequential(
            nn.Linear(self.F * len(Nl), 64),
            nn.ReLU(),
            nn.Linear(64,16)
        ).to(device)
        
        self.color_MLP = nn.Sequential(
            nn.Linear(27 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        ).to(device)

    def positional_encoding(self, x):
        out = [x]
        for i in range(self.L):
            out.append(torch.sin(2 ** i * x))
            out.append(torch.cos(2 ** i * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        x = x / self.aabb_scale
        mask = (x[:, 0].abs() < .5) & (x[:, 1].abs() < .5) & (x[:, 2].abs() < .5)
        x += 0.5 # x in [0,1]^3
        
        color = torch.zeros((x.shape[0], 3), device=x.device)
        log_sigma = torch.zeros((x.shape[0]), device=x.device) - 10000 
        features = torch.empty((x[mask].shape[0], self.F * len(self.Nl)), device=x.device)
        
        for i, N in enumerate(self.Nl):
            #computing vertices, bilinear interpolation
            floor = torch.floor(x[mask] * N)    
            ceil = torch.ceil(x[mask] * N)
            num_vertices = 8
            vertices = torch.zeros((x[mask].shape[0], num_vertices, 3), dtype = torch.int64, device=x.device)
            
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat((ceil[:, 0, None], floor[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 2] = torch.cat((floor[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 3] = torch.cat((floor[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 4] = torch.cat((floor[:, 0, None], ceil[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 5] = torch.cat((ceil[:, 0, None], floor[:, 1, None], ceil[:, 2, None]), dim=1)
            vertices[:, 3] = torch.cat((ceil[:, 0, None], ceil[:, 1, None], floor[:, 2, None]), dim=1)
            vertices[:, 7] = ceil
            
            #hashing
            a = vertices[:, :, 0] = self.pi1
            b = vertices[:, :, 1] = self.pi2
            c = vertices[:, :, 2] = self.pi3
            hash_x = torch.remainder(torch.bitwise_xor(torch.bitwise_xor(a, b), c), self.T)

            