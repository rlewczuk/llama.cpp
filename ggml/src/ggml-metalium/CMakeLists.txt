find_package(tt-nn REQUIRED)

message(STATUS "Metalium/TTNN found.")

ggml_add_backend_library(ggml-metalium
    ggml-metalium.cpp
    ggml-metalium-context.cpp
    ggml-metalium-device.cpp
    ggml-metalium-ops.cpp
    ggml-metalium-util.cpp
)

target_include_directories(ggml-metalium PRIVATE
    # Parent GGML stuff
    ..
)

target_link_libraries(ggml-metalium PRIVATE
    TT::Metalium
    TTNN::TTNN
    ggml-base
)


target_compile_definitions(
    ggml-metalium PRIVATE
    FMT_HEADER_ONLY
    SPDLOG_FMT_EXTERNAL
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation")
endif()

target_precompile_headers(ggml-metalium PRIVATE metalium-pch.hpp)
target_compile_features(ggml-metalium PRIVATE cxx_std_20)
