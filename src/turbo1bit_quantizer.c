/**
 * turbo1bit_quantizer.c — Core TurboQuant quantization algorithms.
 *
 * Ports of turboquant/quantizer.py Algorithm 1 (MSE) and Algorithm 2 (Prod).
 */

#include "turbo1bit_quantizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Bit-packing ─────────────────────────────────────────────────────

static int effective_bits(int bits) {
    // 3-bit stored as 4-bit (matching Python implementation)
    if (bits == 3) return 4;
    return bits;
}

static int vals_per_byte(int bits) {
    int eb = effective_bits(bits);
    return 8 / eb;
}

int t1b_packed_len(int d, int bits) {
    int vpb = vals_per_byte(bits);
    return (d + vpb - 1) / vpb;
}

void t1b_pack_indices(const uint8_t *indices, int d, int bits, uint8_t *packed) {
    int eb = effective_bits(bits);
    int vpb = vals_per_byte(bits);
    int packed_d = t1b_packed_len(d, bits);
    memset(packed, 0, packed_d);

    for (int i = 0; i < d; i++) {
        int byte_idx = i / vpb;
        int pos_in_byte = i % vpb;
        packed[byte_idx] |= (indices[i] & ((1 << eb) - 1)) << (pos_in_byte * eb);
    }
}

void t1b_unpack_indices(const uint8_t *packed, int d, int bits, uint8_t *indices) {
    int eb = effective_bits(bits);
    int vpb = vals_per_byte(bits);
    uint8_t mask = (1 << eb) - 1;

    for (int i = 0; i < d; i++) {
        int byte_idx = i / vpb;
        int pos_in_byte = i % vpb;
        indices[i] = (packed[byte_idx] >> (pos_in_byte * eb)) & mask;
    }
}

int t1b_sign_packed_len(int d) {
    return (d + 7) / 8;
}

void t1b_pack_signs(const uint8_t *signs, int d, uint8_t *packed) {
    int packed_d = t1b_sign_packed_len(d);
    memset(packed, 0, packed_d);
    for (int i = 0; i < d; i++) {
        if (signs[i]) {
            packed[i / 8] |= (1 << (i % 8));
        }
    }
}

void t1b_unpack_signs(const uint8_t *packed, int d, float *signs_pm1) {
    for (int i = 0; i < d; i++) {
        int bit = (packed[i / 8] >> (i % 8)) & 1;
        signs_pm1[i] = bit ? 1.0f : -1.0f;
    }
}

// ── Quantizer state ─────────────────────────────────────────────────

struct t1b_quantizer {
    int                     dim;
    int                     bits;       // total bits (for Prod: MSE uses bits-1)
    int                     mse_bits;   // bits - 1
    t1b_rotation           *rot;
    const struct t1b_codebook *cb;      // codebook for mse_bits
    float                   qjl_scale;  // sqrt(pi/2) / dim

    // Scratch buffers (avoid repeated malloc)
    float  *scratch_rotated;   // dim floats
    float  *scratch_residual;  // dim floats
    float  *scratch_projected; // dim floats
    uint8_t *scratch_indices;  // dim uint8
    uint8_t *scratch_signs;    // dim uint8
};

t1b_quantizer * t1b_quantizer_create(int dim, int bits, int layer_idx) {
    if (bits < 2) return NULL;

    t1b_quantizer *q = (t1b_quantizer *)calloc(1, sizeof(t1b_quantizer));
    if (!q) return NULL;

    q->dim = dim;
    q->bits = bits;
    q->mse_bits = bits - 1;
    q->qjl_scale = sqrtf((float)M_PI / 2.0f) / (float)dim;

    q->rot = t1b_rotation_create(dim, layer_idx);
    q->cb = t1b_get_codebook(dim, q->mse_bits);

    if (!q->rot || !q->cb) {
        t1b_quantizer_free(q);
        return NULL;
    }

    // Allocate scratch buffers
    q->scratch_rotated   = (float  *)malloc(dim * sizeof(float));
    q->scratch_residual  = (float  *)malloc(dim * sizeof(float));
    q->scratch_projected = (float  *)malloc(dim * sizeof(float));
    q->scratch_indices   = (uint8_t *)malloc(dim * sizeof(uint8_t));
    q->scratch_signs     = (uint8_t *)malloc(dim * sizeof(uint8_t));

    if (!q->scratch_rotated || !q->scratch_residual || !q->scratch_projected ||
        !q->scratch_indices || !q->scratch_signs) {
        t1b_quantizer_free(q);
        return NULL;
    }

    return q;
}

void t1b_quantizer_free(t1b_quantizer *q) {
    if (!q) return;
    t1b_rotation_free(q->rot);
    free(q->scratch_rotated);
    free(q->scratch_residual);
    free(q->scratch_projected);
    free(q->scratch_indices);
    free(q->scratch_signs);
    free(q);
}

// ── MSE quantize/dequantize (internal) ──────────────────────────────

static void mse_quantize(const t1b_quantizer *q, const float *x,
                         uint8_t *packed_indices, float *out_norm) {
    int d = q->dim;
    const struct t1b_codebook *cb = q->cb;
    float *rotated = q->scratch_rotated;
    uint8_t *indices = q->scratch_indices;

    // Compute L2 norm
    float norm = 0.0f;
    for (int i = 0; i < d; i++) {
        norm += x[i] * x[i];
    }
    norm = sqrtf(norm);
    *out_norm = norm;

    // Normalize to unit sphere
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    float unit[d]; // VLA (safe for d <= 256)
    for (int i = 0; i < d; i++) {
        unit[i] = x[i] * inv_norm;
    }

    // Apply rotation: rotated = unit @ Pi^T
    t1b_rotate_forward(q->rot, unit, rotated);

    // Quantize each coordinate via searchsorted on decision boundaries
    // decision_boundaries = boundaries[1:-1], length = n_clusters - 1
    int n_boundaries = cb->n_clusters - 1;
    for (int i = 0; i < d; i++) {
        float val = rotated[i];
        // Binary search in decision boundaries
        int idx = 0;
        for (int b = 0; b < n_boundaries; b++) {
            if (val >= cb->boundaries[b + 1]) {
                idx = b + 1;
            } else {
                break;
            }
        }
        indices[i] = (uint8_t)idx;
    }

    // Bit-pack
    t1b_pack_indices(indices, d, q->mse_bits, packed_indices);
}

static void mse_dequantize(const t1b_quantizer *q,
                           const uint8_t *packed_indices, float norm,
                           float *out) {
    int d = q->dim;
    const struct t1b_codebook *cb = q->cb;
    uint8_t *indices = q->scratch_indices;
    float *rotated = q->scratch_rotated;

    // Unpack indices
    t1b_unpack_indices(packed_indices, d, q->mse_bits, indices);

    // Look up centroids
    for (int i = 0; i < d; i++) {
        rotated[i] = cb->centroids[indices[i]];
    }

    // Inverse rotation: out = rotated @ Pi
    t1b_rotate_backward(q->rot, rotated, out);

    // Rescale by norm
    for (int i = 0; i < d; i++) {
        out[i] *= norm;
    }
}

// ── Prod quantize/dequantize ────────────────────────────────────────

void t1b_quantize_prod(const t1b_quantizer *q, const float *x, t1b_prod_quantized *prod_q) {
    int d = q->dim;
    float *residual = q->scratch_residual;
    float *projected = q->scratch_projected;
    uint8_t *signs = q->scratch_signs;

    prod_q->d = d;
    prod_q->mse_bits = q->mse_bits;

    // Stage 1: MSE quantize at (bits-1)
    mse_quantize(q, x, prod_q->mse_indices, &prod_q->norm);

    // Reconstruct MSE approximation to compute residual
    float x_hat[d]; // VLA
    mse_dequantize(q, prod_q->mse_indices, prod_q->norm, x_hat);

    // Compute residual: r = x - x_hat
    for (int i = 0; i < d; i++) {
        residual[i] = x[i] - x_hat[i];
    }

    // Residual norm
    float rnorm = 0.0f;
    for (int i = 0; i < d; i++) {
        rnorm += residual[i] * residual[i];
    }
    prod_q->residual_norm = sqrtf(rnorm);

    // Stage 2: QJL projection on residual
    t1b_qjl_project(q->rot, residual, projected);

    // Pack sign bits
    for (int i = 0; i < d; i++) {
        signs[i] = (projected[i] > 0.0f) ? 1 : 0;
    }
    t1b_pack_signs(signs, d, prod_q->qjl_signs);
}

void t1b_dequantize_prod(const t1b_quantizer *q, const t1b_prod_quantized *prod_q, float *out) {
    int d = q->dim;
    float signs_pm1[d]; // VLA

    // Stage 1: MSE dequantize
    mse_dequantize(q, prod_q->mse_indices, prod_q->norm, out);

    // Stage 2: QJL dequantize
    t1b_unpack_signs(prod_q->qjl_signs, d, signs_pm1);

    // x_qjl = S^T @ signs * (qjl_scale * residual_norm)
    // S^T @ signs = for each output coord i: sum_j S[j][i] * sign[j]
    const float *S = t1b_get_qjl_matrix(q->rot);
    float scale = q->qjl_scale * prod_q->residual_norm;

    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += signs_pm1[j] * S[j * d + i]; // S^T[i][j] = S[j][i]
        }
        out[i] += sum * scale;
    }
}

float t1b_attention_score(const t1b_quantizer *q,
                          const float *query,
                          const float *q_sketched,
                          const t1b_prod_quantized *key_q) {
    int d = q->dim;

    // Stage 1: MSE contribution — <query, k_mse>
    float k_mse[d]; // VLA
    mse_dequantize(q, key_q->mse_indices, key_q->norm, k_mse);

    float score_mse = 0.0f;
    for (int i = 0; i < d; i++) {
        score_mse += query[i] * k_mse[i];
    }

    // Stage 2: QJL contribution — (qjl_scale * ||r||) * <q_sketch, signs>
    float signs_pm1[d]; // VLA
    t1b_unpack_signs(key_q->qjl_signs, d, signs_pm1);

    // If q_sketched not provided, compute it
    float local_sketch[d]; // VLA
    if (!q_sketched) {
        t1b_sketch_query(q, query, local_sketch);
        q_sketched = local_sketch;
    }

    float score_qjl = 0.0f;
    for (int i = 0; i < d; i++) {
        score_qjl += q_sketched[i] * signs_pm1[i];
    }
    score_qjl *= q->qjl_scale * key_q->residual_norm;

    return score_mse + score_qjl;
}

void t1b_sketch_query(const t1b_quantizer *q, const float *query, float *q_sketched) {
    // q_sketched = query @ S^T
    t1b_qjl_project(q->rot, query, q_sketched);
}

// ── Value group quantization ────────────────────────────────────────

int t1b_value_packed_len(int d, int bits) {
    if (bits == 2) return d / 4;
    if (bits == 4) return d / 2;
    return d;
}

void t1b_quantize_values(const float *v, int d, int bits, int group_size,
                         t1b_value_quantized *vq) {
    int n_groups = d / group_size;
    int n_levels = (1 << bits) - 1;

    vq->d = d;
    vq->bits = bits;
    vq->group_size = group_size;
    vq->n_groups = n_groups;

    // For each group, compute min/max and quantize
    uint8_t indices[d]; // VLA — temp uncompressed

    for (int g = 0; g < n_groups; g++) {
        int offset = g * group_size;
        float vmin = v[offset];
        float vmax = v[offset];

        for (int i = 1; i < group_size; i++) {
            float val = v[offset + i];
            if (val < vmin) vmin = val;
            if (val > vmax) vmax = val;
        }

        float scale = (vmax - vmin) / (float)n_levels;
        if (scale < 1e-10f) scale = 1e-10f;

        vq->scales[g] = scale;
        vq->zeros[g] = vmin;

        for (int i = 0; i < group_size; i++) {
            float normalized = (v[offset + i] - vmin) / scale;
            int q_val = (int)(normalized + 0.5f); // round
            if (q_val < 0) q_val = 0;
            if (q_val > n_levels) q_val = n_levels;
            indices[offset + i] = (uint8_t)q_val;
        }
    }

    // Bit-pack
    if (bits == 2) {
        // 4 values per byte: a | (b<<2) | (c<<4) | (d<<6)
        int packed_len = d / 4;
        for (int i = 0; i < packed_len; i++) {
            int base = i * 4;
            vq->data[i] = indices[base]
                        | (indices[base + 1] << 2)
                        | (indices[base + 2] << 4)
                        | (indices[base + 3] << 6);
        }
    } else if (bits == 4) {
        // 2 values per byte: a | (b<<4)
        int packed_len = d / 2;
        for (int i = 0; i < packed_len; i++) {
            int base = i * 2;
            vq->data[i] = indices[base] | (indices[base + 1] << 4);
        }
    } else {
        memcpy(vq->data, indices, d);
    }
}

void t1b_dequantize_values(const t1b_value_quantized *vq, float *out) {
    int d = vq->d;
    int bits = vq->bits;
    int gs = vq->group_size;

    // Unpack to per-element values
    uint8_t indices[d]; // VLA

    if (bits == 2) {
        int packed_len = d / 4;
        for (int i = 0; i < packed_len; i++) {
            uint8_t byte = vq->data[i];
            int base = i * 4;
            indices[base + 0] = byte & 0x03;
            indices[base + 1] = (byte >> 2) & 0x03;
            indices[base + 2] = (byte >> 4) & 0x03;
            indices[base + 3] = (byte >> 6) & 0x03;
        }
    } else if (bits == 4) {
        int packed_len = d / 2;
        for (int i = 0; i < packed_len; i++) {
            uint8_t byte = vq->data[i];
            int base = i * 2;
            indices[base + 0] = byte & 0x0F;
            indices[base + 1] = (byte >> 4) & 0x0F;
        }
    } else {
        memcpy(indices, vq->data, d);
    }

    // Dequantize: val = index * scale + zero
    for (int g = 0; g < vq->n_groups; g++) {
        int offset = g * gs;
        float scale = vq->scales[g];
        float zero = vq->zeros[g];
        for (int i = 0; i < gs; i++) {
            out[offset + i] = (float)indices[offset + i] * scale + zero;
        }
    }
}
