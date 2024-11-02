#include "utils.h"
#include "helper_math.cuh"

__global__ void ray_aabb_intersect_kernel(
    const float *rays_o,
    const float *rays_d,
    const float *centers,
    const float *half_sizes,
    const int max_hits,
    float *hits_t,
    long *hits_voxel_idx,
    int *hit_cnt
){
    
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






