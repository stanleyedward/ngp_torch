// define morton code encoding/decoding
#include "helper_math.cuh"
#include "utils.h"
#include "pcg32.cuh"
#include <torch/extension.h>


torch::Tensor morton3D_cu(const torch::Tensor coords){
    return coords;
}

torch::Tensor morton3D_invert_cu(const torch::Tensor indices){
    return indices;
}


