/**
 * turbo1bit_quantizer.h — TurboQuant MSE and inner-product-unbiased quantizers.
 *
 * Algorithm 1 (MSE): Rotate, quantize each coordinate via Lloyd-Max codebook, bit-pack.
 * Algorithm 2 (Prod): MSE at (b-1) bits + QJL sign projection on residual.
 *
 * The Prod quantizer preserves inner products in expectation, making it suitable
 * for attention key compression where we need <query, key> estimates.
 */

#ifndef TURBO1BIT_QUANTIZER_H
#define TURBO1BIT_QUANTIZER_H

#include "turbo1bit_codebook.h"
#include "turbo1bit_rotation.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Bit-packing utilities ───────────────────────────────────────────

// Pack indices (0..2^bits-1) into bytes. Returns packed length.
// packed must be pre-allocated to at least t1b_packed_len(d, bits) bytes.
int  t1b_packed_len(int d, int bits);
void t1b_pack_indices(const uint8_t *indices, int d, int bits, uint8_t *packed);
void t1b_unpack_indices(const uint8_t *packed, int d, int bits, uint8_t *indices);

// Pack sign bits (0 or 1) into bytes, 8 signs per byte.
int  t1b_sign_packed_len(int d);
void t1b_pack_signs(const uint8_t *signs, int d, uint8_t *packed);
void t1b_unpack_signs(const uint8_t *packed, int d, float *signs_pm1);  // outputs {-1, +1}

// ── MSE Quantized representation ───────────────────────────────────

typedef struct {
    uint8_t *indices;       // bit-packed, length = t1b_packed_len(d, bits)
    float    norm;          // L2 norm of original vector
    int      bits;          // quantization bits
    int      d;             // dimension
} t1b_mse_quantized;

// ── Prod Quantized representation ──────────────────────────────────

typedef struct {
    uint8_t *mse_indices;   // bit-packed MSE indices at (bits-1)
    uint8_t *qjl_signs;     // bit-packed sign bits, length = t1b_sign_packed_len(d)
    float    residual_norm; // L2 norm of residual after MSE reconstruction
    float    norm;          // L2 norm of original vector
    int      mse_bits;      // bits used for MSE stage (= total_bits - 1)
    int      d;             // dimension
} t1b_prod_quantized;

// ── Per-layer quantizer state ──────────────────────────────────────

typedef struct t1b_quantizer t1b_quantizer;

// Create quantizer for a given dimension, total bits, and layer index.
// bits must be >= 2 (1 bit for MSE + 1 for QJL).
t1b_quantizer * t1b_quantizer_create(int dim, int bits, int layer_idx);
void            t1b_quantizer_free(t1b_quantizer *q);

// Quantize a single vector x[d] -> prod_quantized.
// Caller must allocate prod_q->mse_indices and prod_q->qjl_signs.
void t1b_quantize_prod(const t1b_quantizer *q, const float *x, t1b_prod_quantized *prod_q);

// Dequantize back to float vector. out must be float[d].
void t1b_dequantize_prod(const t1b_quantizer *q, const t1b_prod_quantized *prod_q, float *out);

// Compute attention score: <query, compressed_key>
// query must be float[d].
// q_sketched is the pre-computed query @ S^T (float[d]), pass NULL to compute on-the-fly.
float t1b_attention_score(const t1b_quantizer *q,
                          const float *query,
                          const float *q_sketched,
                          const t1b_prod_quantized *key_q);

// Pre-compute query sketch for batch scoring: q_sketched = query @ S^T
void t1b_sketch_query(const t1b_quantizer *q, const float *query, float *q_sketched);

// ── Value group quantization ───────────────────────────────────────

typedef struct {
    uint8_t *data;          // bit-packed quantized values
    float   *scales;        // per-group scales, length = n_groups
    float   *zeros;         // per-group zeros, length = n_groups
    int      bits;          // 2 or 4
    int      d;             // dimension
    int      group_size;    // elements per group
    int      n_groups;      // d / group_size
} t1b_value_quantized;

// Quantize value vector v[d] with group quantization.
// Caller must allocate vq->data, vq->scales, vq->zeros.
void t1b_quantize_values(const float *v, int d, int bits, int group_size,
                         t1b_value_quantized *vq);

// Dequantize value vector back to float. out must be float[d].
void t1b_dequantize_values(const t1b_value_quantized *vq, float *out);

// Packed data length for value quantization
int t1b_value_packed_len(int d, int bits);

#ifdef __cplusplus
}
#endif

#endif // TURBO1BIT_QUANTIZER_H
