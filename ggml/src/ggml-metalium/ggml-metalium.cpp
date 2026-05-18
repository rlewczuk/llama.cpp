#include "ggml-metalium.h"
#include "ggml-metalium-device.h"

#include "ggml-backend.h"
#include "ggml-metalium-context.h"
#include "ggml-metalium.h"

#include <cstdlib>
#include <mutex>

#include <fmt/base.h>

GGML_BACKEND_API bool ggml_backend_is_metalium(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metalium_guid());
}

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_metalium_buffer_type(int device_id) {
    ggml_backend_reg_t reg = ggml_backend_metalium_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, device_id);
    return ggml_backend_dev_buffer_type(dev);
}

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metalium_reg()
{
    static ggml_backend_reg reg;
    static std::once_flag once;
    std::call_once(once, [&]() {
        if(getenv("TT_METAL_HOME") == NULL) {
            fmt::println(stderr, "The TT_METAL_HOME environment variables must be set to use the Metalium backend");
            abort();
        }
        // TODO: Support multiple devices (TT supports mesh configuration so it's going to be tricky)
        // but for now we just work on 1 device at a time
        static std::unique_ptr<ggml_backend_metalium_reg_context> ctx = std::make_unique<ggml_backend_metalium_reg_context>();
        // TODO: Register multiple mesh devices when non-owning TT discovery is available.
        const size_t num_devices = 1;//tt::tt_metal::GetNumAvailableDevices();
        ctx->devices.reserve(num_devices);
        for(size_t device_id = 0; device_id < num_devices; device_id++) {
            ggml_backend_metalium_device_context * dev_ctx = new ggml_backend_metalium_device_context;
            dev_ctx->device_id = device_id;
            dev_ctx->name = "METALIUM" + std::to_string(device_id);
            dev_ctx->description = "Tenstorrent Metalium device";

            ggml_backend_dev_t dev = new ggml_backend_device {
                .iface = ggml_backend_metalium_device_interface,
                .reg = &reg,
                .context = dev_ctx
            };
            ctx->devices.push_back(dev);
            g_backend_device_context_holder.push_back(std::unique_ptr<ggml_backend_metalium_device_context>(dev_ctx));
            g_backend_device_holder.push_back(std::unique_ptr<ggml_backend_device>(dev));
        }

        reg = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .interface   = */ ggml_backend_metalium_reg_interface,
            /* .context     = */ ctx.get()
        };
    });
    return &reg;
}
