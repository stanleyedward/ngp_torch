import torch
import rendering 


class RayAABBIntersector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return rendering.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)

class TruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))