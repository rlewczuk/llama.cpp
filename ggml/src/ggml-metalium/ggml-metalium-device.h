#pragma once

#include "ggml-backend.h"
#include "ggml.h"

bool ggml_backend_is_metalium(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metalium_reg(void);
