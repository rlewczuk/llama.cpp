#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>

namespace ttggml {
using namespace ttnn;

struct SoftMaxOperation {
    static ttnn::Tensor invoke(const Tensor& a, float scale = 1.f);
    static ttnn::Tensor invoke(const Tensor& a, const Tensor& mask, float scale = 1.f);
};
ttnn::Tensor soft_max(const Tensor& a, float scale = 1.f);
ttnn::Tensor soft_max(const Tensor& a, const Tensor& mask, float scale = 1.f);
}
