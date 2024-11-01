// cpp functions that call the kernel
// https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

#include "utils.h"
#include <torch/extension.h>

torch::Tensor morton3D(const torch::Tensor coords){
    CHECK_INPUT(coords);
    
    return morton3D_cu(coords);
}


torch::Tensor morton3D_invert(const torch::Tensor indices){
    CHECK_INPUT(indices);
   
    return morton3D_invert_cu(indices);
}

void packbits(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield){

    CHECK_INPUT(density_grid);
    CHECK_INPUT(density_bitfield);
    
    return packbits_cu(density_grid, density_threshold, density_bitfield);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("morton3D", &morton3D);
  m.def("packbits", &packbits);
  m.def("morton3D_invert", &morton3D_invert);
}