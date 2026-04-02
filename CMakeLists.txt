cmake_minimum_required(VERSION 3.16)
project(Turbo1Bit C)

set(CMAKE_C_STANDARD 11)

# ── Turbo1Bit core library ──────────────────────────────────────────

add_library(turbo1bit STATIC
    src/turbo1bit_codebook.c
    src/turbo1bit_rotation.c
    src/turbo1bit_quantizer.c
    src/turbo1bit_kv_cache.c
)
target_include_directories(turbo1bit PUBLIC src)
target_link_libraries(turbo1bit PRIVATE m)

if(APPLE)
    target_link_libraries(turbo1bit PRIVATE "-framework Accelerate")
endif()

# ── Metal acceleration (macOS only) ─────────────────────────────────

if(APPLE)
    enable_language(OBJC)
    add_library(turbo1bit-metal STATIC src/turbo1bit_metal.m)
    target_include_directories(turbo1bit-metal PUBLIC src)
    target_link_libraries(turbo1bit-metal PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
    set_source_files_properties(src/turbo1bit_metal.m PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
endif()

# ── Benchmark executables ───────────────────────────────────────────

add_executable(turbo1bit-bench tools/turbo1bit/turbo1bit_bench.c)
target_link_libraries(turbo1bit-bench PRIVATE turbo1bit m)
target_include_directories(turbo1bit-bench PRIVATE src)

add_executable(turbo1bit-stress tools/turbo1bit/turbo1bit_stress.c)
target_link_libraries(turbo1bit-stress PRIVATE turbo1bit m)
target_include_directories(turbo1bit-stress PRIVATE src)

if(APPLE)
    target_link_libraries(turbo1bit-bench PRIVATE "-framework Accelerate")
    target_link_libraries(turbo1bit-stress PRIVATE "-framework Accelerate")
endif()
