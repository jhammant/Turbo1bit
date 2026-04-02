/**
 * turbo1bit_metal.h — Metal GPU acceleration for Turbo1Bit KV cache operations.
 *
 * Provides GPU-accelerated versions of the key operations:
 *   - Attention scoring against compressed keys
 *   - Value dequantization
 *   - Fused decode attention with online softmax
 *   - Matrix-vector multiply for rotation/projection
 */

#ifndef TURBO1BIT_METAL_H
#define TURBO1BIT_METAL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque Metal context handle
typedef struct t1b_metal_ctx t1b_metal_ctx;

// Initialize Metal context (loads shaders, creates pipeline states).
// Returns NULL if Metal is not available.
t1b_metal_ctx * t1b_metal_init(void);

// Free Metal context.
void t1b_metal_free(t1b_metal_ctx *ctx);

// Check if Metal acceleration is available.
bool t1b_metal_available(void);

// ── GPU Operations ──────────────────────────────────────────────────

// Compute MSE attention scores on GPU.
// query_rot: pre-rotated query [head_dim]
// mse_packed: bit-packed MSE indices [n_tokens * packed_dim]
// norms: key norms [n_tokens]
// scores_out: output scores [n_tokens]
void t1b_metal_mse_score(
    t1b_metal_ctx *ctx,
    const float *query_rot,
    const uint8_t *mse_packed,
    const float *norms,
    float *scores_out,
    uint32_t n_tokens,
    uint32_t head_dim,
    uint32_t packed_dim,
    uint32_t bits);

// Add QJL score contribution on GPU.
void t1b_metal_qjl_score(
    t1b_metal_ctx *ctx,
    const float *q_sketch,
    const uint8_t *qjl_packed,
    const float *residual_norms,
    float *scores_inout,
    uint32_t n_tokens,
    uint32_t head_dim,
    uint32_t sign_packed_dim);

// Fused decode attention on GPU (single head).
void t1b_metal_fused_attn(
    t1b_metal_ctx *ctx,
    const float *query_rot,
    const float *q_sketch,
    const uint8_t *mse_packed,
    const uint8_t *qjl_packed,
    const float *key_norms,
    const float *residual_norms,
    const uint8_t *val_packed,
    const float *val_scales,
    const float *val_zeros,
    const float *buf_keys,
    const float *buf_values,
    float *output,
    uint32_t n_compressed,
    uint32_t n_buffered,
    uint32_t head_dim,
    float attn_scale);

// In-place value quantize-dequantize on FP16 data (group quantization round-trip).
// Simulates compression: each group is quantized to `bits` then immediately dequantized.
// data is FP16 (half), n_elements total, group_size per group.
void t1b_metal_value_quant_dequant(
    t1b_metal_ctx *ctx,
    void *data_fp16,        // FP16 data, modified in-place
    uint32_t n_elements,
    uint32_t group_size,
    uint32_t bits);

// GPU matrix-vector multiply: y = x @ M^T
void t1b_metal_matvec(
    t1b_metal_ctx *ctx,
    const float *x,
    const float *M,
    float *y,
    uint32_t dim);

#ifdef __cplusplus
}
#endif

#endif // TURBO1BIT_METAL_H
