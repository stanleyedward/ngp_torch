import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import rendering
from custom_functions import RayAABBIntersector, RayMarcher

MAX_SAMPLES = 1024 # fixed!


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat(
        (
            torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
            accumulated_transmittance[:, :-1],
        ),
        dim=-1,
    )


# the paper uses ray marching and an occupancy grid, used volumetric rendering here
def render_rays(model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(
        ray_origins.shape[0], nb_bins
    )

    # sampling across each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)

    # add noise
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u

    delta = torch.cat(
        (
            t[:, 1:] - t[:, :-1],
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),
        ),
        -1,
    )

    # compute the 3d points along ech ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_directions = ray_directions.expand(
        nb_bins, ray_directions.shape[0], 3
    ).transpose(0, 1)

    color, sigma = model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    alpha = 1 - torch.exp(-sigma.reshape(x.shape[:-1]) * delta)  # [batchsize, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2
    ) * alpha.unsqueeze(2)
    c = (weights * color.reshape(x.shape)).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)

def render(model, rays, **kwargs):
    rays_o = rays[:, 0:3].contiguous()
    rays_d = rays[:, 3:6].contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        results[k] = v.cpu() if kwargs.get('to_cpu', False) else v
    return results

@torch.inference_mode()
def __render_rays_test():
    raise NotImplementedError

@torch.autocast(device_type='cuda')
def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    results = {}
    rays_a, xyzs, dirs, deltas, ts = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield, model.scale,
            kwargs.get('exp_step_factor', 0.), True, model.grid_size, MAX_SAMPLES)
        
    sigmas, rgbs = model(xyzs, dirs)
    
    raise NotImplementedError()
    return results
    