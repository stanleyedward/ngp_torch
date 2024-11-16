// cpp functions that call the kernel
// https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

#include "utils.h"
#include <torch/extension.h>

torch::Tensor morton3D(const torch::Tensor coords)
{
    CHECK_INPUT(coords);

    return morton3D_cu(coords);
}

torch::Tensor morton3D_invert(const torch::Tensor indices)
{
    CHECK_INPUT(indices);

    return morton3D_invert_cu(indices);
}

void packbits(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield)
{

    CHECK_INPUT(density_grid);
    CHECK_INPUT(density_bitfield);

    return packbits_cu(density_grid, density_threshold, density_bitfield);
}

std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits)
{
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(half_sizes);
    return ray_aabb_intersect_cu(rays_o, rays_d, centers, half_sizes, max_hits);
}

std::vector<torch::Tensor> raymarching_train(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const bool perturb,
    const int grid_size,
    const int max_samples)
{
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(density_bitfield);

    return raymarching_train_cu(
        rays_o, rays_d, hits_t, density_bitfield, scale, exp_step_factor,
        perturb, grid_size, max_samples);
}

std::vector<torch::Tensor> raymarching_test(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const int grid_size,
    const int max_samples,
    const int N_samples)
{
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(density_bitfield);

    return raymarching_test_cu(
        rays_o, rays_d, hits_t, alive_indices, density_bitfield, scale, exp_step_factor,
        grid_size, max_samples, N_samples);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("morton3D", &morton3D);
    m.def("morton3D_invert", &morton3D_invert);
    m.def("packbits", &packbits);

    m.def("ray_aabb_intersect", &ray_aabb_intersect);

    m.def("raymarching_train", &raymarching_train);
    m.def("raymarching_test", &raymarching_test);
}