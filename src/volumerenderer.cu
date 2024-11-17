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
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0))
        return;

    const int ray_idx = rays_a[n][0];
    const int start_idx = rays_a[n][1];
    const int N_samples = rays_a[n][2];

    // no hits occured
    if (N_samples == 0 || start_idx + N_samples >= sigmas.size(0))
    {
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

    while (samples < N_samples)
    {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s] * deltas[s]);
        const scalar_t w = a * T;

        r += w * rgbs[s][0];
        g += w * rgbs[s][1];
        b += w * rgbs[s][2];
        d += w * ts[s];
        op += w;
        T *= 1.0f - a;

        if (T <= T_threshold)
            break; // ray has enough opacity
        samples++;
    }

    rgb[ray_idx][0] = r + rgb_bg[0] * (1 - op);
    rgb[ray_idx][1] = g + rgb_bg[1] * (1 - op);
    rgb[ray_idx][2] = b + rgb_bg[2] * (1 - op);
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
                                               rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));
    return {opacity, depth, rgb};
}

template <typename scalar_t>
__global__ void composite_train_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dopacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_ddepth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgb,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> rgb_bg,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigmas,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgbs,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_drgb_bg)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0))
        return;

    const int ray_idx = rays_a[n][0];
    const int start_idx = rays_a[n][1];
    const int N_samples = rays_a[n][2];

    if (N_samples == 0 || start_idx + N_samples >= sigmas.size(0))
    { // no hit
        dL_drgb_bg[0] = dL_drgb[ray_idx][0];
        dL_drgb_bg[1] = dL_drgb[ray_idx][1];
        dL_drgb_bg[2] = dL_drgb[ray_idx][2];
        return;
    }
    // front to back compositing
    int samples = 0;
    scalar_t R = rgb[ray_idx][0];
    scalar_t G = rgb[ray_idx][1];
    scalar_t B = rgb[ray_idx][2];
    scalar_t O = opacity[ray_idx];
    scalar_t D = depth[ray_idx];
    scalar_t T = 1.0f;
    scalar_t r = 0.0f;
    scalar_t g = 0.0f;
    scalar_t b = 0.0f;
    scalar_t op = 0.0f;
    scalar_t t = 0.0f;
    scalar_t d = 0.0f;

    while (samples < N_samples)
    {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s] * deltas[s]);
        const scalar_t w = a * T;

        r += w * rgbs[s][0];
        g += w * rgbs[s][1];
        b += w * rgbs[s][2];
        d += w * ts[s];
        op += w;
        T *= 1.0f - a;

        // compute gradients by math...
        dL_drgbs[s][0] = dL_drgb[ray_idx][0] * w;
        dL_drgbs[s][1] = dL_drgb[ray_idx][1] * w;
        dL_drgbs[s][2] = dL_drgb[ray_idx][2] * w;

        dL_dsigmas[s] = deltas[s] * (dL_drgb[ray_idx][0] * (rgbs[s][0] * T - (R - r)) +
                                     dL_drgb[ray_idx][1] * (rgbs[s][1] * T - (G - g)) +
                                     dL_drgb[ray_idx][2] * (rgbs[s][2] * T - (B - b)) +
                                     dL_dopacity[ray_idx] * (1 - O) +
                                     dL_ddepth[ray_idx] * (t * T - (D - d)));

        dL_drgb_bg[0] = dL_drgb[ray_idx][0] * (1 - O);
        dL_drgb_bg[1] = dL_drgb[ray_idx][1] * (1 - O);
        dL_drgb_bg[2] = dL_drgb[ray_idx][2] * (1 - O);

        if (T <= T_threshold)
            break; // ray has enough opacity
        samples++;
    }
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
    const float T_threshold)
{
    const int N = sigmas.size(0);
    const int N_rays = rays_a.size(0);

    auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    auto dL_drgbs = torch::zeros({N, 3}, sigmas.options());
    auto dL_drgb_bg = torch::zeros({3}, sigmas.options());

    const int numThreadsPerBlock = 256;
    const int numBlocks = (N_rays + numThreadsPerBlock - 1) / numThreadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_train_bw_cu",
                                        ([&]
                                         { composite_train_bw_kernel<scalar_t><<<numBlocks, numThreadsPerBlock>>>(
                                               dL_dopacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               dL_ddepth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               dL_drgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                               sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                               deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               rays_a.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                                               opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                               rgb_bg.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               T_threshold,
                                               dL_dsigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               dL_drgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                               dL_drgb_bg.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()); }));

    return {dL_dsigmas, dL_drgbs, dL_drgb_bg};
}

template <typename scalar_t>
__global__ void composite_test_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> hits_t,
    torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> alive_indices,
    const scalar_t T_threshold,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> N_eff_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> depth,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb)
{
    
}
void composite_test_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor rgb)
{
    const int N_rays = alive_indices.size(0);

    const int threads = 256, blocks = (N_rays + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "composite_test_fw_cu",
                                        ([&]
                                         { composite_test_fw_kernel<scalar_t><<<blocks, threads>>>(
                                               sigmas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                               rgbs.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                               deltas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                               ts.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                               hits_t.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                               alive_indices.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                                               T_threshold,
                                               N_eff_samples.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                               opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                               rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));
}