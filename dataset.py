from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import os

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
    
def get_ray_directions(Height:int , Width:int, K_intrinsics):
    pass