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
        self.aabb_scale = aabb_scale
        
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
    