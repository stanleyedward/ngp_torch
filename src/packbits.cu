#include "utils.h"

#include <torch/extension.h>

__global__ void packbits_kernel(
    const float* density_grid,
    const int N,
    const float density_threshold,
    uint8_t* density_bitfield
) {
    // parallel per byte
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Point to the start of the current 8-element chunk
    const float* chunk_start = density_grid + (n * 8);
    uint8_t bits = 0;

    #pragma unroll 8  // unroll 8 times
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (chunk_start[i] > density_threshold) ? ((uint8_t)1 << i) : 0;
    }
    density_bitfield[n] = bits;
}

void packbits_cuda(
    const torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
) {
    // Input validation
    TORCH_CHECK(density_grid.dtype() == torch::kFloat, "density_grid must be float32");
    TORCH_CHECK(density_bitfield.dtype() == torch::kUInt8, "density_bitfield must be uint8");
    TORCH_INTERNAL_ASSERT(density_grid.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(density_bitfield.device().type() == at::DeviceType::CUDA);
    
    // Ensure input is contiguous
    at::Tensor density_grid_contig = density_grid.contiguous();
    
    // Calculate N (number of bytes in bitfield)
    const int N = density_bitfield.size(0);
    TORCH_CHECK(density_grid_contig.numel() == N * 8, 
                "density_grid size must be 8 times the density_bitfield size");
    
    const float* density_grid_ptr = density_grid_contig.data_ptr<float>();
    uint8_t* density_bitfield_ptr = density_bitfield.data_ptr<uint8_t>();
    
    const int numThreadsPerBlock = 256;
    const int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    
    packbits_kernel<<<numBlocks, numThreadsPerBlock>>>(
        density_grid_ptr,
        N,
        density_threshold,
        density_bitfield_ptr
    );
}