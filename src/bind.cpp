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


TORCH_LIBRARY(rendering, m) {
  m.def("morton3D(Tensor a) -> Tensor");
  m.def("morton3D_invert(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(rendering, CPU, m) {
  m.impl("morton3D", &morton3D);
  m.impl("morton3D_invert", &morton3D_invert);
}

// TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
//   m.impl("mymuladd", &mymuladd_cuda);
// }