#include "utils.h"
#include "helper_math.cuh"


__device__ __forceinline__ float2 _ray_aabb_intersect(
    const float3 ray_o,
    const float3 inv_d,
    const float3 center,
    const float3 half_size
){

    const float3 t_min = (center-half_size-ray_o)*inv_d;
    const float3 t_max = (center+half_size-ray_o)*inv_d;

    const float3 _t1 = fminf(t_min, t_max);
    const float3 _t2 = fmaxf(t_min, t_max);
    const float near = fmaxf(fmaxf(_t1.x, _t1.y), _t1.z);
    const float far = fminf(fminf(_t2.x, _t2.y), _t2.z);

    if (near > far) return make_float2(-1.0f, -1.0f); // no intersection
    return make_float2(near, far);
}

__global__ void ray_aabb_intersect_kernel(
    const int N_rays,
    const int N_voxels,
    const float *rays_o,
    const float *rays_d,
    const float *centers,
    const float *half_sizes,
    const int max_hits,
    float *hits_t,
    long *hits_voxel_idx,
    int *hit_cnt
){
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (r>=N_rays || v>=N_voxels) return;
    const float3 ray_o = make_float3(rays_o[r*3+0], rays_o[r*3+1], rays_o[r*3+2]);
    const float3 ray_d = make_float3(rays_d[r*3+0], rays_d[r*3+1], rays_d[r*3+2]);
    const float3 inv_d = 1.0f/ray_d;
    const float3 center = make_float3(centers[v*3+0], centers[v*3+1], centers[v*3+2]);
    const float3 half_size = make_float3(half_sizes[v*3+0], half_sizes[v*3+1], half_sizes[v*3+2]);

    const float2 t1t2 = _ray_aabb_intersect(ray_o, inv_d, center, half_size);

    if (t1t2.y > 0) { // if ray hits the voxel
        const int cnt = atomicAdd(&hit_cnt[r], 1);
        if (cnt < max_hits) {
            hits_t[r*max_hits*2 + cnt*2] = fmaxf(t1t2.x, 0.0f); //store near point
            hits_t[r*max_hits*2 + cnt*2 + 1] = t1t2.y; //store far point
            hits_voxel_idx[r*max_hits + cnt] = v;
        }
    }
}

std::vector<torch::Tensor> ray_aabb_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits){

        const int N_rays = rays_o.size(0);
        const int N_voxels = centers.size(0);

        //contig
        torch::Tensor rays_o_contig = rays_o.contiguous();
        torch::Tensor rays_d_contig = rays_d.contiguous();
        torch::Tensor centers_contig = centers.contiguous();
        torch::Tensor half_sizes_contig = half_sizes.contiguous();


        torch::Tensor hits_t = 
            torch::zeros({N_rays, max_hits, 2}, rays_o.options()) -1;
        torch::Tensor hits_voxel_idx = 
            torch::zeros({N_rays, max_hits},
            torch::dtype(torch::kLong).device(rays_o.device())) -1;
        torch::Tensor hit_cnt = 
            torch::zeros({N_rays}, torch::dtype(torch::kInt32).device(rays_o.device()));


        //pointers
        const float *rays_o_contig_ptr = rays_o_contig.data_ptr<float>();
        const float *rays_d_contig_ptr = rays_d_contig.data_ptr<float>();
        const float *centers_contig_ptr = centers_contig.data_ptr<float>();
        const float *half_sizes_contig_ptr = half_sizes_contig.data_ptr<float>();

        float *hits_t_ptr = hits_t.data_ptr<float>();
        long *hits_voxel_idx_ptr = hits_voxel_idx.data_ptr<long>();
        int *hit_cnt_ptr = hit_cnt.data_ptr<int>();

        //launch kernel
        const dim3 numThreadsPerBlock(256, 1, 1);
        const dim3 numBlocks((N_rays + numThreadsPerBlock.x -1) / numThreadsPerBlock.x, 
                             (N_voxels + numThreadsPerBlock.y -1) / numThreadsPerBlock.y,
                             1);
        ray_aabb_intersect_kernel<<<numBlocks, numThreadsPerBlock>>>(
            N_rays,
            N_voxels,
            rays_o_contig_ptr,
            rays_d_contig_ptr,
            centers_contig_ptr,
            half_sizes_contig_ptr,
            max_hits,
            hits_t_ptr,
            hits_voxel_idx_ptr,
            hit_cnt_ptr
        );

        //sort interesction s from near to far based on t1
        auto hits_order = std::get<1>(torch::sort(hits_t.index({"...", 0})));
        hits_voxel_idx = torch::gather(hits_voxel_idx, 1, hits_order);
        hits_t = torch::gather(hits_t, 1, hits_order.unsqueeze(-1).tile({1, 1, 2}));

        return {hit_cnt, hits_t, hits_voxel_idx};
}






