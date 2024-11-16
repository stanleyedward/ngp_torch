#include "utils.h"
#include "pcg32.cuh"
#include "helper_math.cuh"
#include "morton.cu"

#define SQRT3 1.73205080757f

inline __host__ __device__ float signf(const float x) { return copysignf(1.0f, x); }

inline __host__ __device__ float calc_dt(
    float t, float exp_step_factor, float scale,
    int max_samples, int grid_size, int cascades){
    return clamp(t*exp_step_factor,
                 SQRT3*2*scale/max_samples,
                 SQRT3*2*(1<<(cascades-1))/grid_size);
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const int cascades) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent; frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, ...
    return fminf(cascades-1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(float dt, int grid_size, int cascades) {
    int exponent; frexpf(dt*2*grid_size, &exponent);
    return fminf(cascades-1, fmaxf(0, exponent));
}


// below code is based on https://github.com/ashawkey/torch-ngp/blob/main/raymarching/src/raymarching.cu
__global__ void raymarching_train_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> hits_t,
    const uint8_t* __restrict__ density_bitfield,
    const int cascades,
    const int grid_size,
    const float scale,
    const float exp_step_factor,
    const bool perturb,
    const int max_samples,
    pcg32 rng,
    int* __restrict__ counter,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> xyzs,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> dirs,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> deltas,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> ts
){
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rays_o.size(0)) return;

    const int grid_size3 = grid_size*grid_size*grid_size;

    const float ox = rays_o[r][0], oy = rays_o[r][1], oz = rays_o[r][2];
    const float dx = rays_d[r][0], dy = rays_d[r][1], dz = rays_d[r][2];
    float t1 = hits_t[r][0], t2 = hits_t[r][1];

    if (perturb && t1>=0) { // only perturb the starting t
        rng.advance(r);
        const float dt = 
            calc_dt(t1, exp_step_factor, scale, max_samples, grid_size, cascades);
        t1 += dt*rng.next_float();
    }

    // first pass: compute the number of samples on the ray
    float t = t1; int N_samples = 0;

    // if t1 < 0 (no hit) this loop will be skipped (N_samples will be 0)
    while (0<=t && t<t2 && N_samples<max_samples){
        const float x = clamp(ox+t*dx, -scale, scale);
        const float y = clamp(oy+t*dy, -scale, scale);
        const float z = clamp(oz+t*dz, -scale, scale);

        const float dt = 
            calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
        const int mip = max(mip_from_pos(x, y, z, cascades),
                            mip_from_dt(dt, grid_size, cascades));

        const float mip_bound = fminf(1<<mip, scale);
        const float mip_bound_inv = 1.0f/mip_bound;

        // round down to nearest grid position
        const int nx = clamp(0.5f*(x*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp(0.5f*(y*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp(0.5f*(z*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);

        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));

        if (occ) {
            t += dt; N_samples++;
        } else { // skip until the next voxel
            // calculate the distance to the next voxel
            float grid_size_inv = 1.0f/grid_size;
            const float tx = (((nx+0.5f*(1+signf(dx)))*grid_size_inv*2-1)*mip_bound-x)/dx;
            const float ty = (((ny+0.5f*(1+signf(dy)))*grid_size_inv*2-1)*mip_bound-y)/dy;
            const float tz = (((nz+0.5f*(1+signf(dz)))*grid_size_inv*2-1)*mip_bound-z)/dz;

            const float t_target = t+fmaxf(0.0f, fminf(tx, fminf(ty, tz))); // the t of the next voxel
            do {
                t += calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
            } while (t < t_target);
        }
    }

    // second pass: write to output
    int start_idx = atomicAdd(counter, N_samples);
    int ray_count = atomicAdd(counter+1, 1);

    rays_a[ray_count][0] = r;
    rays_a[ray_count][1] = start_idx; rays_a[ray_count][2] = N_samples;

    if (N_samples==0 || start_idx+N_samples>=xyzs.size(0)) return;

    t = t1; int samples = 0;

    while (t<t2 && samples<N_samples){
        const float x = clamp(ox+t*dx, -scale, scale);
        const float y = clamp(oy+t*dy, -scale, scale);
        const float z = clamp(oz+t*dz, -scale, scale);

        const float dt = 
            calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
        const int mip = max(mip_from_pos(x, y, z, cascades),
                            mip_from_dt(dt, grid_size, cascades));

        const float mip_bound = fminf(1<<mip, scale);
        const float mip_bound_inv = 1.0f/mip_bound;

        // round down to nearest grid position
        const int nx = clamp(0.5f*(x*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp(0.5f*(y*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp(0.5f*(z*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);

        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));

        if (occ) {
            const int s = start_idx + samples;
            xyzs[s][0] = x; xyzs[s][1] = y; xyzs[s][2] = z;
            dirs[s][0] = dx; dirs[s][1] = dy; dirs[s][2] = dz;
            ts[s] = t;
            deltas[s] = dt; // interval for volume rendering integral
            t += dt; samples++;
        } else { // skip until the next voxel
            // calculate the distance to the next voxel
            const float grid_size_inv = 1.0f/grid_size;
            const float tx = (((nx+0.5f*(1+signf(dx)))*grid_size_inv*2-1)*mip_bound-x)/dx;
            const float ty = (((ny+0.5f*(1+signf(dy)))*grid_size_inv*2-1)*mip_bound-y)/dy;
            const float tz = (((nz+0.5f*(1+signf(dz)))*grid_size_inv*2-1)*mip_bound-z)/dz;

            const float t_target = t+fmaxf(0.0f, fminf(tx, fminf(ty, tz))); // the t of the next voxel
            do {
                t += calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
            } while (t < t_target);
        }
    }
}


std::vector<torch::Tensor> raymarching_train_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const bool perturb,
    const int grid_size,
    const int max_samples
){
    const int N_rays = rays_o.size(0);
    const int cascades = density_bitfield.size(0);

    // count the number of samples and the number of rays processed
    auto counter = torch::zeros({2}, torch::dtype(torch::kInt32).device(rays_o.device()));
    // ray attributes: ray_idx, start_idx, N_samples
    auto rays_a = torch::zeros({N_rays, 3},
                        torch::dtype(torch::kInt32).device(rays_o.device()));
    auto xyzs = torch::zeros({N_rays*max_samples, 3}, rays_o.options());
    auto dirs = torch::zeros({N_rays*max_samples, 3}, rays_o.options());
    auto deltas = torch::zeros({N_rays*max_samples}, rays_o.options());
    auto ts = torch::zeros({N_rays*max_samples}, rays_o.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    pcg32 rng = pcg32{(uint64_t)42}; // hard coded random seed
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.type(), "raymarching_train_cu", 
    ([&] {
        raymarching_train_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            hits_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            density_bitfield.data_ptr<uint8_t>(),
            cascades,
            grid_size,
            scale,
            exp_step_factor,
            perturb,
            max_samples,
            rng,
            counter.data_ptr<int>(),
            rays_a.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            xyzs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            deltas.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            ts.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
        );
    }));

    return {rays_a, xyzs, dirs, deltas, ts, counter};
}


__global__ void raymarching_test_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> hits_t,
    const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> alive_indices,
    const uint8_t* __restrict__ density_bitfield,
    const int cascades,
    const int grid_size,
    const float scale,
    const float exp_step_factor,
    const int N_samples,
    const int max_samples,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> xyzs,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> dirs,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> deltas,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ts,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> N_eff_samples
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0)) return;

    const size_t r = alive_indices[n]; // ray index
    const int grid_size3 = grid_size*grid_size*grid_size;

    const float ox = rays_o[r][0], oy = rays_o[r][1], oz = rays_o[r][2];
    const float dx = rays_d[r][0], dy = rays_d[r][1], dz = rays_d[r][2];

    float t = hits_t[r][0], t2 = hits_t[r][1];
    int s = 0;

    while (t<t2 && s<N_samples){
        const float x = clamp(ox+t*dx, -scale, scale);
        const float y = clamp(oy+t*dy, -scale, scale);
        const float z = clamp(oz+t*dz, -scale, scale);

        const float dt = 
            calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
        const int mip = max(mip_from_pos(x, y, z, cascades),
                            mip_from_dt(dt, grid_size, cascades));

        const float mip_bound = fminf(1<<mip, scale);
        const float mip_bound_inv = 1.0f/mip_bound;

        // round down to nearest grid position
        const int nx = clamp(0.5f*(x*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int ny = clamp(0.5f*(y*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);
        const int nz = clamp(0.5f*(z*mip_bound_inv+1)*grid_size, 0.0f, grid_size-1.0f);

        const uint32_t idx = mip*grid_size3 + __morton3D(nx, ny, nz);
        const bool occ = density_bitfield[idx/8] & (1<<(idx%8));

        if (occ) {
            xyzs[n][s][0] = x; xyzs[n][s][1] = y; xyzs[n][s][2] = z;
            dirs[n][s][0] = dx; dirs[n][s][1] = dy; dirs[n][s][2] = dz;
            ts[n][s] = t;
            deltas[n][s] = dt; // interval for volume rendering integral
            t += dt;
            hits_t[r][0] = t; // modify the starting point for the next marching
            s++;
        } else { // skip until the next voxel
            // calculate the distance to the next voxel
            const float grid_size_inv = 1.0f/grid_size;
            const float tx = (((nx+0.5f*(1+signf(dx)))*grid_size_inv*2-1)*mip_bound-x)/dx;
            const float ty = (((ny+0.5f*(1+signf(dy)))*grid_size_inv*2-1)*mip_bound-y)/dy;
            const float tz = (((nz+0.5f*(1+signf(dz)))*grid_size_inv*2-1)*mip_bound-z)/dz;

            const float t_target = t+fmaxf(0.0f, fminf(tx, fminf(ty, tz))); // the t of the next voxel
            do {
                t += calc_dt(t, exp_step_factor, scale, max_samples, grid_size, cascades);
            } while (t < t_target);
        }
    }
    N_eff_samples[n] = s; // effective samples that hit occupied region (<=N_samples)
}


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
    const int N_samples
){
    const int N_rays = alive_indices.size(0);
    const int cascades = density_bitfield.size(0);

    auto xyzs = torch::zeros({N_rays, N_samples, 3}, rays_o.options());
    auto dirs = torch::zeros({N_rays, N_samples, 3}, rays_o.options());
    auto deltas = torch::zeros({N_rays, N_samples}, rays_o.options());
    auto ts = torch::zeros({N_rays, N_samples}, rays_o.options());
    auto N_eff_samples = torch::zeros({N_rays},
                            torch::dtype(torch::kInt32).device(rays_o.device()));

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.type(), "raymarching_test_cu", 
    ([&] {
        raymarching_test_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            hits_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            alive_indices.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
            density_bitfield.data_ptr<uint8_t>(),
            cascades,
            grid_size,
            scale,
            exp_step_factor,
            N_samples,
            max_samples,
            xyzs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            dirs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            deltas.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ts.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            N_eff_samples.packed_accessor32<int, 1, torch::RestrictPtrTraits>()
        );
    }));

    return {xyzs, dirs, deltas, ts, N_eff_samples};
}
