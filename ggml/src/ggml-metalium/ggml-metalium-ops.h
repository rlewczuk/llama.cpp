#pragma once

#include "ggml-backend.h"
#include "ggml.h"

struct ggml_backend_metalium_context;

enum ggml_status ggml_backend_metalium_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);

bool ggml_backend_metalium_can_mul_mat(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_cpy(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_set(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_set_rows(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_get_rows(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_concat(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_softmax(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_outer_product(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_glu(const struct ggml_tensor * dst);
bool ggml_backend_metalium_can_sum(const struct ggml_tensor * dst);
