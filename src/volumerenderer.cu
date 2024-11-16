#include "utils.h"

std::vector<torch::Tensor> composite_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor rgb_bg,
    const float T_threshold)
{
    const int N_rays = rays_a.size(0);

    auto opacity = torch::zeros({N_rays}, sigmas.options());
    auto depth = torch::zeros({N_rays}, sigmas.options());
    auto rgb = torch::zeros({N_rays, 3}, sigmas.options());

    return {opacity, depth, rgb};
}