// declare cuda kernels here so the bind.cpp can find them
#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TORCH(x) TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA)
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_TORCH(x)

torch::Tensor morton3D_cu(const torch::Tensor coords);
torch::Tensor morton3D_invert_cu(const torch::Tensor indices);

void packbits_cu(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
);

std::vector<torch::Tensor> ray_aabb_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits
);
