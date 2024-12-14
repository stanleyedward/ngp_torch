import torch
import rendering
from einops import rearrange
from custom_functions import RayAABBIntersector, RayMarcher, VolumeRenderer

MAX_SAMPLES = 1024 


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

@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):

    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device =rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = 0
    alive_indices = torch.arange(N_rays, device=device)

    while samples < MAX_SAMPLES:
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), 1)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            rendering.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield,
                                  model.scale, kwargs.get('exp_step_factor', 0.),
                                  model.grid_size, MAX_SAMPLES, N_samples)
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        _sigmas, _rgbs = model(xyzs[valid_mask], dirs[valid_mask])
        sigmas[valid_mask] = _sigmas.float()
        rgbs[valid_mask] = _rgbs.float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        rendering.composite_test_fw(
            sigmas, rgbs, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rgb)
        alive_indices = alive_indices[alive_indices>=0]

    rgb_bg = torch.ones(3, device=device)
    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb + rgb_bg*rearrange(1-opacity, 'n -> n 1')

    return results

@torch.autocast(device_type='cuda')
def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    results = {}
    rays_a, xyzs, dirs, deltas, ts = RayMarcher.apply(
        rays_o, rays_d, hits_t[:, 0], model.density_bitfield, model.scale,
        kwargs.get('exp_step_factor', 0.), True, model.grid_size, MAX_SAMPLES)
        
    sigmas, rgbs = model(xyzs, dirs)
    rgb_bg = torch.ones(3, device=rays_o.device)
    results['opacity'], results['depth'], results['rgb'] = VolumeRenderer.apply(
        sigmas, rgbs, deltas, ts, rays_a, rgb_bg, kwargs.get('T_threshold', 1e-4))
    return results
    