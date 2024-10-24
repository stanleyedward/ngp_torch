from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import os
import torch
from kornia import create_meshgrid
from glob import glob
from tqdm import tqdm
from PIL import Image
import einops

class NSVFDataset(Dataset):
    def __init__(self, root_dir:str, split:str="train", downsample:float=1.0, 
                    batch_size:int=8192, **kwargs):
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
        f_x *= downsample; f_y *= downsample
        K_intrinsics = np.float32([[f_x, 0, w/2],
                                 [0, f_y, h/2],
                                 [0, 0,     1]])
        
        self.image_size = (w, h)
        self.directions = get_ray_directions(h, w, K_intrinsics)
        
        if split.startswith('train'):
            rays_train = self.read_meta('train')
            self.rays = torch.cat(list(rays_train.values()))
        else:
            self.rays = self.read_meta(split)
            
    def read_meta(self, split):
        rays = {}
        
        if split == 'train': 
            prefix = '0_'
        elif split == 'val':
            prefix = '1_'
        elif 'Synthetic' in self.root_dir:
            prefix = '2_'
        elif split == 'test':
            prefix = '1_' 
            
        imgs = sorted(glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
        poses = sorted(glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))
        
        print(f"[INFO] len:{len(imgs)} split: {split}")
        for idx, (img, pose) in enumerate(tqdm(zip(imgs, poses))):
            cam2world = np.loadtxt(pose)[:3]
            cam2world[:, 1:3] *= -1
            cam2world[:, 3] -= self.shift
            cam2world[:, 3] /= self.scale
            rays_o, rays_d = get_rays(self.directions, torch.FloatTensor(cam2world))
            
            img = Image.open(img)
            img = img.resize(self.image_size, Image.LANCZOS)
            img = self.transform(img) #[c,h,w]
            img = einops.rearrange(img, 'c h w -> (h w) c')
            rays[idx] = torch.cat([rays_o, rays_d, img], dim=1)
            
            return rays

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split.startswith('train'):
            return 1000 * self.batch_size
        return len(self.rays)

    def __getitem__(self, index):
        if self.split.startswith('train'):
            index = np.random.randint(len(self.rays))
            sample = {
                'rays': self.rays[index, :6],
                'rgb': self.rays[index, 6:9],
                'idx': index
                }
        else:
            sample = {
                'rays': self.rays[index][:, :6],
                'rgb': self.rays[index][:, 6:9]
                     }
        return sample
    
        
def get_rays(directions, cam2world):
    rays_directions = directions @ cam2world[:, :3].T
    rays_directions /= torch.norm(rays_directions, dim=-1, keepdim=True)
    rays_origins = cam2world[:, 3].expand(rays_directions.shape) #[h*w,3]
    return rays_origins, rays_directions


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
    
    
    
    