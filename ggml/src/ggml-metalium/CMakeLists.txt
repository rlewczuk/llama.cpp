find_package(tt-nn REQUIRED)

message(STATUS "Metalium/TTNN found.")

ggml_add_backend_library(ggml-metalium
    ggml-metalium.cpp
    utils.cpp
    rope.cpp
    mul_mat.cpp
    soft_max.cpp
)

target_include_directories(ggml-metalium PRIVATE
    # Parent GGML stuff
    ..
)

target_link_libraries(ggml-metalium PRIVATE
    TT::Metalium
    TTNN::TTNN
    ggml-base
    ggml-cpu
)

if(GGML_METALIUM_EMBED_KERNELS)
    add_library(ggml-metalium-kernels STATIC)
    set(GGML_METALIUM_EMBED_KERNELS
            rope_neox_reader
            rope_neox_compute
            rope_neox_writer
            rope_normal_reader
            rope_normal_compute
            rope_normal_writer
            mul_mat_reader
            mul_mat_compute
            mul_mat_writer
            soft_max_reader
            soft_max_writer
            soft_max_compute
    )
    include(cmake/add_kernel.cmake)
    foreach(kernel IN ITEMS ${GGML_METALIUM_EMBED_KERNELS})
        add_kernel(ggml-metalium-kernels ${kernel})
    endforeach()
    register_kernels(ggml-metalium "${GGML_METALIUM_EMBED_KERNELS}")
    target_link_libraries(ggml-metalium PRIVATE ggml-metalium-kernels)
    target_compile_definitions(ggml-metalium PRIVATE GGML_METALIUM_EMBED_KERNELS)
    set_target_properties(ggml-metalium-kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(ggml-metalium-kernels PRIVATE -Wno-missing-declarations)
endif()

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
