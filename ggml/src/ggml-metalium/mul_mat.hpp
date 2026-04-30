#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>

namespace ttggml {
using namespace ttnn;

struct MulMatOperation {
    static ttnn::Tensor invoke(const Tensor& a, const Tensor& b, bool high_percision = false);
};

/**
 * Implements GGML's MUL_MAT operation wich computes b @ aT
 */
ttnn::Tensor mul_mat(const Tensor& a, const Tensor& b, bool high_percision = false);
}
