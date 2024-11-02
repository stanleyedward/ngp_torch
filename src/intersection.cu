#include "utils.h"
#include "helper_math.cuh"

__global__ void ray_aabb_intersect_kernel(){
    
}

torch::Tensor ray_aabb_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits){

}






