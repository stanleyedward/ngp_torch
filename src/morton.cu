// define morton code encoding/decoding
#include "helper_math.cuh"
#include "utils.h"
#include "pcg32.cuh"
#include <torch/extension.h>

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t xx = __expand_bits(x);
    uint32_t yy = __expand_bits(y);
    uint32_t zz = __expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
    x = x & 0x49249249;
    x = (x | (x >> 2)) & 0xc30c30c3;
    x = (x | (x >> 4)) & 0x0f00f00f;
    x = (x | (x >> 8)) & 0xff0000ff;
    x = (x | (x >> 16)) & 0x0000ffff;
    return x;
}

__global__ void morton3D_kernel(
    int N,
    const int *coords, // [N, 3]
    int *indices       // [N]
)
{
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N)
        return;

    // access coords as a flattened array, with stride of 3
    const int x = coords[n * 3 + 0];
    const int y = coords[n * 3 + 1];
    const int z = coords[n * 3 + 2];

    indices[n] = __morton3D(x, y, z);
}

at::Tensor morton3D_cu(const at::Tensor coords)
{
    TORCH_CHECK(coords.dim() == 2, "coords must be 2D");
    TORCH_CHECK(coords.size(1) == 3, "coords must have 3 coordinates");
    TORCH_CHECK(coords.dtype() == at::kInt);
    TORCH_INTERNAL_ASSERT(coords.device().type() == at::DeviceType::CUDA);

    const int N = coords.size(0);
    at::Tensor coords_contig = coords.contiguous();
    at::Tensor indices = torch::zeros({N}, coords.options());

    // raw pointers
    const int *coords_ptr = coords_contig.data_ptr<int>();
    int *indices_ptr = indices.data_ptr<int>();

    const int numThreadsPerBlock = 256;
    const int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    morton3D_kernel<<<numBlocks, numThreadsPerBlock>>>(N, coords_ptr, indices_ptr);

    return indices;
}

__global__ void morton3D_invert_kernel(
    int N,
    const int *indices, // [N]
    int *coords         // [N, 3]
)
{
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N)
        return;

    const int ind = indices[n];
    // write to coords as a flattened array with stride of 3
    coords[n * 3 + 0] = __morton3D_invert(ind >> 0);
    coords[n * 3 + 1] = __morton3D_invert(ind >> 1);
    coords[n * 3 + 2] = __morton3D_invert(ind >> 2);
}

at::Tensor morton3D_invert_cu(const at::Tensor indices)
{
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");
    TORCH_CHECK(indices.dtype() == at::kInt);
    TORCH_INTERNAL_ASSERT(indices.device().type() == at::DeviceType::CUDA);

    const int N = indices.size(0);

    at::Tensor indices_contig = indices.contiguous();
    at::Tensor coords = torch::zeros({N, 3}, indices.options());

    // raw pointers
    const int *indices_ptr = indices_contig.data_ptr<int>();
    int *coords_ptr = coords.data_ptr<int>();

    const int numThreadsPerBlock = 256;
    const int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    morton3D_invert_kernel<<<numBlocks, numThreadsPerBlock>>>(N, indices_ptr, coords_ptr);

    return coords;
}