// declare cuda kernels here so the bind.cpp can find them
#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_TORCH(x) TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA)
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_TORCH(x)

torch::Tensor morton3D_cu(const torch::Tensor coords);
torch::Tensor morton3D_invert_cu(const torch::Tensor indices);

void packbits_cu(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield);

std::vector<torch::Tensor> ray_aabb_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits);

std::vector<torch::Tensor> raymarching_train_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const bool perturb,
    const int grid_size,
    const int max_samples);

std::vector<torch::Tensor> raymarching_test_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const int grid_size,
    const int max_samples,
    const int N_samples);

std::vector<torch::Tensor> composite_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor rgb_bg,
    const float T_threshold
);

std::vector<torch::Tensor> composite_train_bw_cu(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddepth,
    const torch::Tensor dL_drgb,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    const torch::Tensor rgb,
    const torch::Tensor rgb_bg,
    const float T_threshold
);
