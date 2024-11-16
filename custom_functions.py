import torch
import rendering 


class TruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))
    
    
class RayAABBIntersector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return rendering.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)


class RayMarcher(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rays_o, rays_d, hits_t, density_bitfield, scale, exp_step_factor, perturb, grid_size, max_samples):
        rays_a, xyzs, dirs, deltas, ts, counter = \
            rendering.raymarching_train(
                rays_o, rays_d, hits_t,
                density_bitfield, scale,
                exp_step_factor, perturb, grid_size, max_samples)
            
        total_samples = counter[0] #total samples for all rays
        #remove redundant outputs
        xyzs = xyzs[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]
        
        return rays_a, xyzs, dirs, deltas, ts
        
