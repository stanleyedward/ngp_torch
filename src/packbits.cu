#include "utils.h"

#include <torch/extension.h>

__global__ void packbits_kernel(
    const float *density_grid,
    const int N,
    const float density_threshold,
    uint8_t *density_bitfield)
{
    // parallel per byte
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N)
        return;

    // Point to the start of the current 8-element chunk
    const float *chunk_start = density_grid + (n * 8);
    uint8_t bits = 0;

#pragma unroll 8 // unroll 8 times
    for (uint8_t i = 0; i < 8; i++)
    {
        bits |= (chunk_start[i] > density_threshold) ? ((uint8_t)1 << i) : 0;
    }
    density_bitfield[n] = bits;
}

void packbits_cuda(
    const torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield)
{

    // Calculate N (number of bytes in bitfield)
    const int N = density_bitfield.size(0);
    const int numThreadsPerBlock = 256;
    const int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    packbits_kernel<<<numBlocks, numThreadsPerBlock>>>(
        density_grid.contiguous().data_ptr<float>(),
        N,
        density_threshold,
        density_bitfield.data_ptr<uint8_t>()

    );
}