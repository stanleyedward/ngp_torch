from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import os
import torch
from kornia import create_meshgrid

class NSVFDataset(Dataset):
    def __init__(self, root_dir:str, split:str="train", downsample:float=1.0, 
                    batch_size:int=32, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.batch_size = batch_size
        
        coords_min, coords_max = np.loadtxt(os.path.join(root_dir, "bbox.txt"))[:6].reshape(2,3)
        self.shift = (coords_max+coords_min)/2
        self.scale = (coords_max-coords_min).max()/2 * 1.05
        self.define_transforms()
        
        with open(os.path.join(root_dir, 'intrinsics.txt')) as f:
            f_x = f_y = float(f.readline().split()[0])
        w = h = int(800*downsample)
        K_intrinsics = np.float([[f_x, 0, w/2],
                                 [0, f_y, h/2],
                                 [0, 0,     1]])
        
        self.image_size = (w, h)
    
    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
def get_ray_directions(height:int , width:int, K_intrinsics):
    grid = create_meshgrid(height=height, width=width, normalized_coordinates=False)[0] #[H,W,2]
    i, j = grid.unbind(-1)
    
    f_x = K_intrinsics[0, 0]
    f_y = K_intrinsics[1, 1]
    c_x = K_intrinsics[0, 2]
    c_y = K_intrinsics[1, 2]
    
    directions = torch.stack([
        (i-c_x+0.5)/f_x,
        -(j-c_y+0.5)/f_y,
        -torch.ones_like(i)]
        , -1)
    
    #flatten
    directions = directions.reshape(-1, 3)
    return directions
    
    
    
    