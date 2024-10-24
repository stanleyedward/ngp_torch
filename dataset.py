import torch
import torch
from kornia import create_meshgrid
import glob
import numpy as np
import os
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as T


class NSVFDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        downsample: float = 1.0,
        batch_size: int = 8192,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.batch_size = batch_size
        self.define_transforms()

        xyz_min, xyz_max = np.loadtxt(os.path.join(root_dir, "bbox.txt"))[:6].reshape(
            2, 3
        )
        self.shift = (xyz_max + xyz_min) / 2
        self.scale = (xyz_max - xyz_min).max() / 2 * 1.05  # Enlarge slightly

        self.load_intrinsics_synthetic(root_dir, downsample)

        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], self.K)

        if split.startswith("train"):
            rays_train = self.read_meta("train")
            self.rays = torch.cat(list(rays_train.values()))
        else:
            self.rays = self.read_meta(split)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def load_intrinsics_synthetic(self, root_dir, downsample):
        with open(os.path.join(root_dir, "intrinsics.txt")) as f:
            fx = fy = float(f.readline().split()[0])
        w = h = int(800 * downsample)
        fx *= downsample
        fy *= downsample
        self.img_wh = (w, h)
        self.K = np.float32([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])

    def read_meta(self, split):
        rays = {}  # {frame_idx: ray tensor}

        prefix = self.get_prefix_for_split(split)
        imgs = sorted(glob.glob(os.path.join(self.root_dir, "rgb", prefix + "*.png")))
        poses = sorted(glob.glob(os.path.join(self.root_dir, "pose", prefix + "*.txt")))

        print(f"Loading {len(imgs)} {split} images ...")
        for idx, (img, pose) in enumerate(tqdm(zip(imgs, poses))):
            c2w = np.loadtxt(pose)[:3]
            c2w[:, 1:3] *= -1  # Adjust orientation
            c2w[:, 3] = (
                c2w[:, 3] - self.shift
            ) / self.scale  # Normalize scene coordinates
            rays_o, rays_d = get_rays(self.directions, torch.FloatTensor(c2w))

            img = Image.open(img).resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            img = rearrange(img, "c h w -> (h w) c")
            img = self.adjust_alpha_channel(img)

            rays[idx] = torch.cat([rays_o, rays_d, img], 1)  # (h*w, 9)

        return rays

    def adjust_alpha_channel(self, img):
        if img.shape[-1] == 4:
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # Blend alpha to RGB
        return img

    def get_prefix_for_split(self, split):
        if split == "train":
            return "0_"
        elif split == "val":
            return "1_"
        elif "Synthetic" in self.root_dir:
            return "2_"
        elif split == "test":
            return "1_"

    def __len__(self):
        if self.split.startswith("train"):
            return 1000 * self.batch_size  # Example: Epoch length for training
        return len(self.rays)

    def __getitem__(self, idx):
        if self.split.startswith("train"):
            idx = np.random.randint(len(self.rays))
            sample = {
                "rays": self.rays[idx, :6],
                "rgb": self.rays[idx, 6:9],
                "idx": idx,
            }
        else:
            sample = {"rays": self.rays[idx][:, :6], "rgb": self.rays[idx][:, 6:9]}
        return sample


def get_ray_directions(H, W, K):

    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]  # (H, W, 2)
    i, j = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = torch.stack(
        [(i - cx + 0.5) / fx, -(j - cy + 0.5) / fy, -torch.ones_like(i)], -1
    )
    # flatten:
    directions = directions.reshape(-1, 3)
    return directions


def get_rays(directions, c2w):
    rays_d = directions @ c2w[:, :3].T  # (H*W, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H*W, 3)

    return rays_o, rays_d
