/**
 * turbo1bit_metal.metal — Metal compute shaders for Turbo1Bit compressed KV cache.
 *
 * Three kernels ported from TurboQuant's Triton implementation:
 *   1. t1b_mse_score:   Compute attention scores against MSE-quantized keys
 *   2. t1b_qjl_score:   Add QJL residual contribution to scores
 *   3. t1b_fused_attn:  Fused decode attention with online softmax
 *
 * Plus helper kernels:
 *   4. t1b_dequant_values: Unpack and dequantize group-quantized values
 *   5. t1b_quantize_keys:  Compress keys during prefill/decode
 */

#include <metal_stdlib>
using namespace metal;

// ── Constants ───────────────────────────────────────────────────────

// Lloyd-Max codebook centroids for d=128
// 2-bit (4 centroids) — used for MSE stage of 3-bit Prod quantizer
constant float centroids_d128_b2[4] = {
    -0.1330401982533685f, -0.039990945215356365f,
     0.039990945215356365f, 0.1330401982533685f
};

// Decision boundaries for d=128, 2-bit
constant float boundaries_d128_b2[3] = {
    -0.08651557173436243f, 0.0f, 0.08651557173436243f
};

// QJL scale constant: sqrt(pi/2) / d
constant float QJL_SCALE_128 = 0.009801714f;  // sqrt(pi/2) / 128

// ── Kernel 1: MSE Score ─────────────────────────────────────────────
//
// Computes: score[t] = sum_j query_rot[j] * centroids[indices[t][j]] * norms[t]
// where query_rot = query @ Pi^T (pre-rotated by host)
//
// Each thread handles one KV token.

kernel void t1b_mse_score(
    device const float    *query_rot    [[buffer(0)]],  // [head_dim]
    device const uint8_t  *mse_packed   [[buffer(1)]],  // [n_tokens, packed_dim]
    device const float    *norms        [[buffer(2)]],  // [n_tokens]
    device       float    *scores       [[buffer(3)]],  // [n_tokens]
    constant     uint     &n_tokens     [[buffer(4)]],
    constant     uint     &head_dim     [[buffer(5)]],
    constant     uint     &packed_dim   [[buffer(6)]],
    constant     uint     &bits         [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_tokens) return;

    uint vals_per_byte = 8 / bits;
    uint mask = (1u << bits) - 1u;

    float dot = 0.0f;
    device const uint8_t *packed = mse_packed + tid * packed_dim;

    for (uint byte_idx = 0; byte_idx < packed_dim; byte_idx++) {
        uint8_t byte_val = packed[byte_idx];
        for (uint v = 0; v < vals_per_byte; v++) {
            uint coord = byte_idx * vals_per_byte + v;
            if (coord >= head_dim) break;

            uint idx = (byte_val >> (v * bits)) & mask;
            // Use the 2-bit codebook (for 3-bit Prod = 2-bit MSE + 1-bit QJL)
            float centroid = centroids_d128_b2[idx];
            dot += query_rot[coord] * centroid;
        }
    }

    scores[tid] = dot * norms[tid];
}

// ── Kernel 2: QJL Score ─────────────────────────────────────────────
//
// Adds QJL residual contribution to existing MSE scores:
//   scores[t] += (sqrt(pi/2)/d) * residual_norms[t] * sum_j q_sketch[j] * sign[t][j]
//
// q_sketch = query @ S^T (pre-computed by host)

kernel void t1b_qjl_score(
    device const float    *q_sketch      [[buffer(0)]],  // [head_dim]
    device const uint8_t  *qjl_packed    [[buffer(1)]],  // [n_tokens, sign_packed_dim]
    device const float    *residual_norms[[buffer(2)]],  // [n_tokens]
    device       float    *scores        [[buffer(3)]],  // [n_tokens] (accumulated)
    constant     uint     &n_tokens      [[buffer(4)]],
    constant     uint     &head_dim      [[buffer(5)]],
    constant     uint     &sign_packed_dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_tokens) return;

    float dot = 0.0f;
    device const uint8_t *packed = qjl_packed + tid * sign_packed_dim;

    for (uint byte_idx = 0; byte_idx < sign_packed_dim; byte_idx++) {
        uint8_t byte_val = packed[byte_idx];
        for (uint b = 0; b < 8; b++) {
            uint coord = byte_idx * 8 + b;
            if (coord >= head_dim) break;

            float sign = ((byte_val >> b) & 1) ? 1.0f : -1.0f;
            dot += q_sketch[coord] * sign;
        }
    }

    scores[tid] += QJL_SCALE_128 * residual_norms[tid] * dot;
}

// ── Kernel 3: Fused Decode Attention ────────────────────────────────
//
// Full fused attention kernel with online softmax:
//   1. Compute MSE + QJL scores per KV token
//   2. Online softmax
//   3. Dequantize values and accumulate weighted output
//
// This kernel processes one head. Each threadgroup handles a block of KV tokens.

#define BLOCK_SIZE 64

kernel void t1b_fused_attn(
    // Query (pre-transformed)
    device const float    *query_rot     [[buffer(0)]],   // [head_dim]
    device const float    *q_sketch      [[buffer(1)]],   // [head_dim]
    // Compressed keys
    device const uint8_t  *mse_packed    [[buffer(2)]],   // [n_tokens, mse_packed_dim]
    device const uint8_t  *qjl_packed    [[buffer(3)]],   // [n_tokens, sign_packed_dim]
    device const float    *key_norms     [[buffer(4)]],   // [n_tokens]
    device const float    *residual_norms[[buffer(5)]],   // [n_tokens]
    // Compressed values
    device const uint8_t  *val_packed    [[buffer(6)]],   // [n_tokens, val_packed_dim]
    device const float    *val_scales    [[buffer(7)]],   // [n_tokens, n_groups]
    device const float    *val_zeros     [[buffer(8)]],   // [n_tokens, n_groups]
    // Output
    device       float    *output        [[buffer(9)]],   // [head_dim]
    // Params
    constant     uint     &n_compressed  [[buffer(10)]],
    constant     uint     &head_dim      [[buffer(11)]],
    constant     uint     &mse_packed_dim[[buffer(12)]],
    constant     uint     &sign_packed_dim[[buffer(13)]],
    constant     uint     &val_packed_dim[[buffer(14)]],
    constant     uint     &n_groups      [[buffer(15)]],
    constant     uint     &group_size    [[buffer(16)]],
    constant     float    &attn_scale    [[buffer(17)]],
    // Buffer KV (full precision, appended after compressed)
    device const float    *buf_keys      [[buffer(18)]],  // [n_buffered, head_dim]
    device const float    *buf_values    [[buffer(19)]],  // [n_buffered, head_dim]
    constant     uint     &n_buffered    [[buffer(20)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tsize [[threads_per_threadgroup]])
{
    // This kernel is dispatched with 1 threadgroup per head.
    // Each thread in the group collaborates on the reduction.

    uint total_tokens = n_compressed + n_buffered;
    if (total_tokens == 0) {
        if (tid < head_dim) output[tid] = 0.0f;
        return;
    }

    // Shared memory for partial results
    threadgroup float shared_max[1];
    threadgroup float shared_sum[1];

    // Phase 1: Compute all scores (each thread handles a subset of tokens)
    // For simplicity, allocate scores in device memory (passed as output temp)

    // Simple single-thread implementation for correctness first
    // TODO: Parallelize across threads in the threadgroup
    if (tid == 0) {
        float m_i = -INFINITY;
        float l_i = 0.0f;

        // Initialize accumulator
        float acc[128]; // max head_dim
        for (uint d = 0; d < head_dim; d++) acc[d] = 0.0f;

        // Process compressed tokens
        for (uint t = 0; t < n_compressed; t++) {
            // MSE score
            float score = 0.0f;
            device const uint8_t *mp = mse_packed + t * mse_packed_dim;
            for (uint byte_idx = 0; byte_idx < mse_packed_dim; byte_idx++) {
                uint8_t bv = mp[byte_idx];
                for (uint v = 0; v < 4; v++) { // 2-bit = 4 per byte
                    uint coord = byte_idx * 4 + v;
                    if (coord >= head_dim) break;
                    uint idx = (bv >> (v * 2)) & 0x3;
                    score += query_rot[coord] * centroids_d128_b2[idx];
                }
            }
            score *= key_norms[t];

            // QJL score
            float qjl_dot = 0.0f;
            device const uint8_t *qp = qjl_packed + t * sign_packed_dim;
            for (uint byte_idx = 0; byte_idx < sign_packed_dim; byte_idx++) {
                uint8_t bv = qp[byte_idx];
                for (uint b = 0; b < 8; b++) {
                    uint coord = byte_idx * 8 + b;
                    if (coord >= head_dim) break;
                    float sign = ((bv >> b) & 1) ? 1.0f : -1.0f;
                    qjl_dot += q_sketch[coord] * sign;
                }
            }
            score += QJL_SCALE_128 * residual_norms[t] * qjl_dot;
            score *= attn_scale;

            // Online softmax update
            float m_new = max(m_i, score);
            float alpha = exp(m_i - m_new);
            float p = exp(score - m_new);
            l_i = l_i * alpha + p;
            for (uint d = 0; d < head_dim; d++) acc[d] *= alpha;

            // Dequantize value and accumulate
            device const uint8_t *vp = val_packed + t * val_packed_dim;
            device const float *vs = val_scales + t * n_groups;
            device const float *vz = val_zeros + t * n_groups;
            for (uint g = 0; g < n_groups; g++) {
                float sc = vs[g];
                float zr = vz[g];
                uint base = g * group_size;
                for (uint i = 0; i < group_size; i += 4) {
                    uint byte_off = (base + i) / 4;
                    uint8_t bv = vp[byte_off];
                    acc[base + i + 0] += p * (float((bv >> 0) & 0x3) * sc + zr);
                    acc[base + i + 1] += p * (float((bv >> 2) & 0x3) * sc + zr);
                    acc[base + i + 2] += p * (float((bv >> 4) & 0x3) * sc + zr);
                    acc[base + i + 3] += p * (float((bv >> 6) & 0x3) * sc + zr);
                }
            }

            m_i = m_new;
        }

        // Process buffered tokens (full precision)
        for (uint t = 0; t < n_buffered; t++) {
            float score = 0.0f;
            device const float *k = buf_keys + t * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                score += query_rot[d] * k[d]; // Note: buf keys are NOT rotated
            }
            score *= attn_scale;

            float m_new = max(m_i, score);
            float alpha = exp(m_i - m_new);
            float p = exp(score - m_new);
            l_i = l_i * alpha + p;
            for (uint d = 0; d < head_dim; d++) acc[d] *= alpha;

            device const float *v = buf_values + t * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                acc[d] += p * v[d];
            }

            m_i = m_new;
        }

        // Normalize
        float inv_l = 1.0f / (l_i + 1e-10f);
        for (uint d = 0; d < head_dim; d++) {
            output[d] = acc[d] * inv_l;
        }
    }
}

// ── Kernel 4: Value Dequantization ──────────────────────────────────
//
// Unpacks and dequantizes group-quantized values.
// Each thread handles one token's value vector.

kernel void t1b_dequant_values(
    device const uint8_t  *packed       [[buffer(0)]],  // [n_tokens, packed_dim]
    device const float    *scales       [[buffer(1)]],  // [n_tokens, n_groups]
    device const float    *zeros        [[buffer(2)]],  // [n_tokens, n_groups]
    device       float    *output       [[buffer(3)]],  // [n_tokens, head_dim]
    constant     uint     &n_tokens     [[buffer(4)]],
    constant     uint     &head_dim     [[buffer(5)]],
    constant     uint     &packed_dim   [[buffer(6)]],
    constant     uint     &n_groups     [[buffer(7)]],
    constant     uint     &group_size   [[buffer(8)]],
    constant     uint     &bits         [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_tokens) return;

    device const uint8_t *p = packed + tid * packed_dim;
    device const float *s = scales + tid * n_groups;
    device const float *z = zeros + tid * n_groups;
    device float *out = output + tid * head_dim;

    if (bits == 2) {
        for (uint g = 0; g < n_groups; g++) {
            float scale = s[g];
            float zero = z[g];
            uint base = g * group_size;

            for (uint i = 0; i < group_size; i += 4) {
                uint byte_idx = (base + i) / 4;
                uint8_t byte_val = p[byte_idx];

                out[base + i + 0] = float((byte_val >> 0) & 0x3) * scale + zero;
                out[base + i + 1] = float((byte_val >> 2) & 0x3) * scale + zero;
                out[base + i + 2] = float((byte_val >> 4) & 0x3) * scale + zero;
                out[base + i + 3] = float((byte_val >> 6) & 0x3) * scale + zero;
            }
        }
    } else if (bits == 4) {
        for (uint g = 0; g < n_groups; g++) {
            float scale = s[g];
            float zero = z[g];
            uint base = g * group_size;

            for (uint i = 0; i < group_size; i += 2) {
                uint byte_idx = (base + i) / 2;
                uint8_t byte_val = p[byte_idx];

                out[base + i + 0] = float(byte_val & 0xF) * scale + zero;
                out[base + i + 1] = float((byte_val >> 4) & 0xF) * scale + zero;
            }
        }
    }
}

// ── Kernel 5: Matrix-Vector Multiply (for rotation) ─────────────────
//
// Computes y = x @ M^T where M is a d x d matrix (rotation or QJL).
// Used for: query rotation (x @ Pi^T) and query sketching (x @ S^T).

kernel void t1b_matvec(
    device const float *x      [[buffer(0)]],  // [dim]
    device const float *M      [[buffer(1)]],  // [dim, dim] row-major
    device       float *y      [[buffer(2)]],  // [dim]
    constant     uint  &dim    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;

    // y[tid] = sum_j x[j] * M[tid][j]  (= x @ M^T row tid)
    float sum = 0.0f;
    device const float *row = M + tid * dim;
    for (uint j = 0; j < dim; j++) {
        sum += x[j] * row[j];
    }
    y[tid] = sum;
}
