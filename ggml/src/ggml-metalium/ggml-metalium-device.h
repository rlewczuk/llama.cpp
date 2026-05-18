#pragma once

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

#include <memory>
#include <vector>

struct ggml_backend_metalium_device_context;

ggml_guid_t ggml_backend_metalium_guid(void);
extern const ggml_backend_reg_i ggml_backend_metalium_reg_interface;
extern const ggml_backend_device_i ggml_backend_metalium_device_interface;
extern std::vector<std::unique_ptr<ggml_backend_device>> g_backend_device_holder;
extern std::vector<std::unique_ptr<ggml_backend_metalium_device_context>> g_backend_device_context_holder;

bool ggml_backend_is_metalium(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metalium_reg(void);
