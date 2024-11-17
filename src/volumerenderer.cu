#include "utils.h"

template <typename scalar_t>
__global__ void composite_train_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> rgb_bg,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0]; 
    const int start_idx = rays_a[n][1];
    const int N_samples = rays_a[n][2];

    //no hits occured
    if(N_samples==0 || start_idx+N_samples>=sigmas.size(0)){ 
        rgb[ray_idx][0] = rgb_bg[0];
        rgb[ray_idx][1] = rgb_bg[1];
        rgb[ray_idx][2] = rgb_bg[2];
        return;
    }
    // front to back compositing
    int samples = 0;
    scalar_t T = 1.0f;
    scalar_t r = 0.0f;
    scalar_t g = 0.0f;
    scalar_t b = 0.0f;
    scalar_t op = 0.0f;
    scalar_t d = 0.0f;

    while (samples < N_samples){
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        r += w*rgbs[s][0];
        g += w*rgbs[s][1];
        b += w*rgbs[s][2];
        d += w*ts[s];
        op+= w;
        T *= 1.0f-a;

        if (T<=T_threshold) break; //ray has enough opacity
        samples ++;
    }

    rgb[ray_idx][0] = r + rgb_bg[0]*(1-op);
    rgb[ray_idx][1] = g + rgb_bg[1]*(1-op);
    rgb[ray_idx][2] = b + rgb_bg[2]*(1-op);
    opacity[ray_idx] = op;
    depth[ray_idx] = d;
}
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

    const int numThreadsPerBlock = 256;
    const int numBlocks = (N_rays + numThreadsPerBlock - 1) / numThreadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_fw_cu",
        ([&]
            { composite_train_fw_kernel<scalar_t><<<numBlocks, numThreadsPerBlock>>>(
                sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                rays_a.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                rgb_bg.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                T_threshold,
                opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
                );
            }));
    return {opacity, depth, rgb};
}

template <typename scalar_t>
__global__ void composite_train_bw_kernel(){

}

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
){

}