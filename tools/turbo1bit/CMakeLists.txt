add_executable(turbo1bit-stress
    turbo1bit_stress.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_codebook.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_rotation.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_quantizer.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_kv_cache.c
)
target_include_directories(turbo1bit-stress PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(turbo1bit-stress PRIVATE m)
if(APPLE)
    target_link_libraries(turbo1bit-stress PRIVATE "-framework Accelerate")
endif()
set_target_properties(turbo1bit-stress PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Metal acceleration library (macOS only)
if(APPLE)
    add_library(turbo1bit-metal STATIC
        ${CMAKE_SOURCE_DIR}/src/turbo1bit_metal.m
    )
    target_include_directories(turbo1bit-metal PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(turbo1bit-metal PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
    target_compile_options(turbo1bit-metal PRIVATE -fobjc-arc)
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/src/turbo1bit_metal.m PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    # Metal shaders compiled at runtime from source (no Metal toolchain needed)
endif()

add_executable(turbo1bit-debug
    turbo1bit_debug.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_codebook.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_rotation.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_quantizer.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_kv_cache.c
)
target_include_directories(turbo1bit-debug PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(turbo1bit-debug PRIVATE m)
if(APPLE)
    target_link_libraries(turbo1bit-debug PRIVATE "-framework Accelerate")
endif()
set_target_properties(turbo1bit-debug PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(turbo1bit-bench
    turbo1bit_bench.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_codebook.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_rotation.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_quantizer.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_kv_cache.c
)

target_include_directories(turbo1bit-bench PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Link math library and Accelerate (macOS) for LAPACK
target_link_libraries(turbo1bit-bench PRIVATE m)
if(APPLE)
    target_link_libraries(turbo1bit-bench PRIVATE "-framework Accelerate")
endif()

set_target_properties(turbo1bit-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
