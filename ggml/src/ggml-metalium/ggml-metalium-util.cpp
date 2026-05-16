#include "ggml-metalium-util.h"
#include "ttnn/tensor/shape/shape.hpp"

bool ggml_tt_tensors_shape_equal(const ggml_tensor* ggtensor, const tt::tt_metal::Tensor& ttensor)
{
    const ttnn::Shape& shape = ttensor.logical_shape();
    for(size_t i = 0; i < std::min<size_t>(GGML_MAX_DIMS, shape.size()); i++) {
        if(ggtensor->ne[GGML_MAX_DIMS - i - 1] != shape[i]) {
            return false;
        }
    }

    if(shape.size() > GGML_MAX_DIMS) {
        for(size_t i = GGML_MAX_DIMS; i < shape.size(); i++) {
            if(shape[i] != 1) {
                return false;
            }
        }
    }
    else if(shape.size() < GGML_MAX_DIMS) {
        for(size_t i = shape.size(); i < GGML_MAX_DIMS; i++) {
            if(ggtensor->ne[GGML_MAX_DIMS - i - 1] != 1) {
                return false;
            }
        }
    }
    return true;
}

void ggml_tt_dump_tensor_data(const ggml_tensor* ggtensor)
{
    std::cerr << "GGML tensor: " << ggtensor->name << "\n"
        << "  type: " << ggml_type_name(ggtensor->type) << "\n"
        << "  ne: " << ggtensor->ne[0] << " " << ggtensor->ne[1] << " " << ggtensor->ne[2] << " " << ggtensor->ne[3] << "\n"
        << "  nb: " << ggtensor->nb[0] << " " << ggtensor->nb[1] << " " << ggtensor->nb[2] << " " << ggtensor->nb[3] << "\n"
        << "  op: " << ggml_op_name(ggtensor->op) << "\n"
        << "  data: " << ggtensor->data << "\n"
        << "  src0: " << ggtensor->src[0] << "\n";
    if(ggtensor->src[0] != nullptr) {
        std::cerr << "    src0->name: " << ggtensor->src[0]->name << "\n"
            << "    src0->type: " << ggml_type_name(ggtensor->src[0]->type) << "\n"
            << "    src0->ne:   " << ggtensor->src[0]->ne[0] << " " << ggtensor->src[0]->ne[1] << " " << ggtensor->src[0]->ne[2] << " " << ggtensor->src[0]->ne[3] << "\n"
            << "    src0->nb:   " << ggtensor->src[0]->nb[0] << " " << ggtensor->src[0]->nb[1] << " " << ggtensor->src[0]->nb[2] << " " << ggtensor->src[0]->nb[3] << "\n"
            << "    src0->op:   " << ggml_op_name(ggtensor->src[0]->op) << "\n"
            << "    src0->data: " << ggtensor->src[0]->data << "\n";
    }
    std::cerr << "  src1: " << ggtensor->src[1] << "\n";
    if(ggtensor->src[1] != nullptr) {
        std::cerr << "    src1->name: " << ggtensor->src[1]->name << "\n"
            << "    src1->type: " << ggml_type_name(ggtensor->src[1]->type) << "\n"
            << "    src1->ne: " << ggtensor->src[1]->ne[0] << " " << ggtensor->src[1]->ne[1] << " " << ggtensor->src[1]->ne[2] << " " << ggtensor->src[1]->ne[3] << "\n"
            << "    src1->nb: " << ggtensor->src[1]->nb[0] << " " << ggtensor->src[1]->nb[1] << " " << ggtensor->src[1]->nb[2] << " " << ggtensor->src[1]->nb[3] << "\n"
            << "    src1->op: " << ggml_op_name(ggtensor->src[1]->op) << "\n"
            << "    src1->data: " << ggtensor->src[1]->data << "\n";
    }
    std::cerr << "  view_src: " << ggtensor->view_src << "\n";
    if(ggtensor->view_src != nullptr) {
        std::cerr << "    view_src->name: " << ggtensor->view_src->name << "\n"
            << "    view_src->type: " << ggml_type_name(ggtensor->view_src->type) << "\n"
            << "    view_src->ne: " << ggtensor->view_src->ne[0] << " " << ggtensor->view_src->ne[1] << " " << ggtensor->view_src->ne[2] << " " << ggtensor->view_src->ne[3] << "\n"
            << "    view_src->nb: " << ggtensor->view_src->nb[0] << " " << ggtensor->view_src->nb[1] << " " << ggtensor->view_src->nb[2] << " " << ggtensor->view_src->nb[3] << "\n"
            << "    view_src->op: " << ggml_op_name(ggtensor->view_src->op) << "\n"
            << "    view_src->data: " << ggtensor->view_src->data << "\n";
    }
}
