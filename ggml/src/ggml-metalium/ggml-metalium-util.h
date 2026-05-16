#pragma once

#include "ggml.h"
#include "ttnn/tensor/tensor.hpp"

bool ggml_tt_tensors_shape_equal(const ggml_tensor* ggtensor, const tt::tt_metal::Tensor& ttensor);
void ggml_tt_dump_tensor_data(const ggml_tensor* ggtensor);
