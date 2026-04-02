/**
 * turbo1bit_kv_cache.c — Compressed KV cache implementation.
 *
 * Manages a hybrid cache with:
 *   - Compressed storage for older tokens (TurboQuant keys + group-quantized values)
 *   - Full-precision buffer for recent tokens
 *   - Automatic flushing when buffer overflows
 */

#include "turbo1bit_kv_cache.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ── Internal structures ─────────────────────────────────────────────

// Compressed token storage for one head
typedef struct {
    uint8_t *mse_indices;     // bit-packed MSE indices
    uint8_t *qjl_signs;      // bit-packed QJL signs
    float    residual_norm;
    float    norm;
} t1b_compressed_key;

typedef struct {
    uint8_t *data;            // bit-packed quantized values
    float   *scales;          // per-group scales
    float   *zeros;           // per-group zeros
} t1b_compressed_value;

struct t1b_kv_cache {
    t1b_kv_cache_config config;

    // Per-head quantizers
    t1b_quantizer **quantizers;  // [n_heads]

    // Compressed storage — dynamically growing arrays
    t1b_compressed_key   *comp_keys;    // [max_compressed * n_heads]
    t1b_compressed_value *comp_values;  // [max_compressed * n_heads]
    int n_compressed;      // number of compressed tokens
    int max_compressed;    // allocated capacity

    // Full-precision buffer for recent tokens
    float *buf_keys;       // [buffer_size, n_heads, head_dim]
    float *buf_values;     // [buffer_size, n_heads, head_dim]
    int    n_buffered;     // number of tokens in buffer

    // Pre-allocated sizes
    int mse_packed_len;    // bytes per key MSE indices
    int sign_packed_len;   // bytes per key QJL signs
    int val_packed_len;    // bytes per value data
    int val_n_groups;      // groups per value
};

// ── Helpers ─────────────────────────────────────────────────────────

static int hd_stride(const t1b_kv_cache *c) {
    return c->config.n_heads * c->config.head_dim;
}

static void alloc_compressed_slot(t1b_kv_cache *c, int token_idx) {
    int nh = c->config.n_heads;
    for (int h = 0; h < nh; h++) {
        int idx = token_idx * nh + h;
        c->comp_keys[idx].mse_indices = (uint8_t *)calloc(c->mse_packed_len, 1);
        c->comp_keys[idx].qjl_signs   = (uint8_t *)calloc(c->sign_packed_len, 1);
        c->comp_values[idx].data   = (uint8_t *)calloc(c->val_packed_len, 1);
        c->comp_values[idx].scales = (float *)calloc(c->val_n_groups, sizeof(float));
        c->comp_values[idx].zeros  = (float *)calloc(c->val_n_groups, sizeof(float));
    }
}

static void grow_compressed(t1b_kv_cache *c, int needed) {
    if (c->n_compressed + needed <= c->max_compressed) return;

    int new_max = c->max_compressed * 2;
    if (new_max < c->n_compressed + needed) {
        new_max = c->n_compressed + needed + 256;
    }

    int nh = c->config.n_heads;
    size_t old_count = (size_t)c->max_compressed * nh;
    size_t new_count = (size_t)new_max * nh;

    t1b_compressed_key *new_keys = (t1b_compressed_key *)realloc(
        c->comp_keys, new_count * sizeof(t1b_compressed_key));
    t1b_compressed_value *new_values = (t1b_compressed_value *)realloc(
        c->comp_values, new_count * sizeof(t1b_compressed_value));

    if (!new_keys || !new_values) {
        // Realloc failed — keep old pointers if one succeeded
        if (new_keys) c->comp_keys = new_keys;
        if (new_values) c->comp_values = new_values;
        return;
    }

    c->comp_keys = new_keys;
    c->comp_values = new_values;

    // Zero-init new slots
    memset(&c->comp_keys[old_count], 0,
           (new_count - old_count) * sizeof(t1b_compressed_key));
    memset(&c->comp_values[old_count], 0,
           (new_count - old_count) * sizeof(t1b_compressed_value));

    c->max_compressed = new_max;
}

static void compress_token(t1b_kv_cache *c, const float *key, const float *value, int comp_idx) {
    int nh = c->config.n_heads;
    int hd = c->config.head_dim;

    alloc_compressed_slot(c, comp_idx);

    for (int h = 0; h < nh; h++) {
        int idx = comp_idx * nh + h;
        const float *k = key + h * hd;
        const float *v = value + h * hd;

        // Compress key via TurboQuantProd
        t1b_prod_quantized pq;
        pq.mse_indices = c->comp_keys[idx].mse_indices;
        pq.qjl_signs   = c->comp_keys[idx].qjl_signs;
        t1b_quantize_prod(c->quantizers[h], k, &pq);
        c->comp_keys[idx].residual_norm = pq.residual_norm;
        c->comp_keys[idx].norm          = pq.norm;

        // Compress value via group quantization
        t1b_value_quantized vq;
        vq.data   = c->comp_values[idx].data;
        vq.scales = c->comp_values[idx].scales;
        vq.zeros  = c->comp_values[idx].zeros;
        t1b_quantize_values(v, hd, c->config.value_bits,
                            c->config.value_group_size, &vq);
    }
}

// ── Public API ──────────────────────────────────────────────────────

t1b_kv_cache * t1b_kv_cache_create(const t1b_kv_cache_config *config) {
    t1b_kv_cache *c = (t1b_kv_cache *)calloc(1, sizeof(t1b_kv_cache));
    if (!c) return NULL;

    c->config = *config;

    // Pre-compute sizes
    c->mse_packed_len = t1b_packed_len(config->head_dim, config->key_bits - 1);
    c->sign_packed_len = t1b_sign_packed_len(config->head_dim);
    c->val_packed_len = t1b_value_packed_len(config->head_dim, config->value_bits);
    c->val_n_groups = config->head_dim / config->value_group_size;

    // Create per-head quantizers
    c->quantizers = (t1b_quantizer **)calloc(config->n_heads, sizeof(t1b_quantizer *));
    if (!c->quantizers) { t1b_kv_cache_free(c); return NULL; }

    for (int h = 0; h < config->n_heads; h++) {
        // Each head in same layer shares the same rotation matrices
        // (matching Python: seed = 42 + layer_idx * 7)
        c->quantizers[h] = t1b_quantizer_create(config->head_dim, config->key_bits, config->layer_idx);
        if (!c->quantizers[h]) { t1b_kv_cache_free(c); return NULL; }
    }

    // Allocate compressed storage — pre-allocate for expected usage
    int init_cap = config->max_seq_len > 0 ? config->max_seq_len : 4096;
    c->max_compressed = init_cap;
    c->comp_keys   = (t1b_compressed_key *)  calloc(init_cap * config->n_heads, sizeof(t1b_compressed_key));
    c->comp_values = (t1b_compressed_value *)calloc(init_cap * config->n_heads, sizeof(t1b_compressed_value));

    // Allocate buffer
    int stride = config->n_heads * config->head_dim;
    c->buf_keys   = (float *)calloc(config->buffer_size * stride, sizeof(float));
    c->buf_values = (float *)calloc(config->buffer_size * stride, sizeof(float));

    if (!c->comp_keys || !c->comp_values || !c->buf_keys || !c->buf_values) {
        t1b_kv_cache_free(c);
        return NULL;
    }

    return c;
}

void t1b_kv_cache_free(t1b_kv_cache *c) {
    if (!c) return;

    if (c->quantizers) {
        for (int h = 0; h < c->config.n_heads; h++) {
            t1b_quantizer_free(c->quantizers[h]);
        }
        free(c->quantizers);
    }

    // Free compressed data
    if (c->comp_keys) {
        int total = c->max_compressed * c->config.n_heads;
        for (int i = 0; i < total; i++) {
            free(c->comp_keys[i].mse_indices);
            free(c->comp_keys[i].qjl_signs);
        }
        free(c->comp_keys);
    }
    if (c->comp_values) {
        int total = c->max_compressed * c->config.n_heads;
        for (int i = 0; i < total; i++) {
            free(c->comp_values[i].data);
            free(c->comp_values[i].scales);
            free(c->comp_values[i].zeros);
        }
        free(c->comp_values);
    }

    free(c->buf_keys);
    free(c->buf_values);
    free(c);
}

void t1b_kv_cache_clear(t1b_kv_cache *c) {
    // Free compressed slot data
    int total = c->n_compressed * c->config.n_heads;
    for (int i = 0; i < total; i++) {
        free(c->comp_keys[i].mse_indices);
        free(c->comp_keys[i].qjl_signs);
        c->comp_keys[i].mse_indices = NULL;
        c->comp_keys[i].qjl_signs = NULL;
        free(c->comp_values[i].data);
        free(c->comp_values[i].scales);
        free(c->comp_values[i].zeros);
        c->comp_values[i].data = NULL;
        c->comp_values[i].scales = NULL;
        c->comp_values[i].zeros = NULL;
    }
    c->n_compressed = 0;
    c->n_buffered = 0;
}

int t1b_kv_cache_seq_len(const t1b_kv_cache *c) {
    return c->n_compressed + c->n_buffered;
}

void t1b_kv_cache_append(t1b_kv_cache *c, const float *key, const float *value) {
    int stride = hd_stride(c);
    int bs = c->config.buffer_size;

    // Add to buffer
    memcpy(c->buf_keys   + c->n_buffered * stride, key,   stride * sizeof(float));
    memcpy(c->buf_values + c->n_buffered * stride, value, stride * sizeof(float));
    c->n_buffered++;

    // Flush if buffer full
    if (c->n_buffered > bs) {
        int n_flush = c->n_buffered - bs;
        grow_compressed(c, n_flush);

        for (int t = 0; t < n_flush; t++) {
            compress_token(c,
                           c->buf_keys   + t * stride,
                           c->buf_values + t * stride,
                           c->n_compressed + t);
        }
        c->n_compressed += n_flush;

        // Shift buffer
        int remaining = c->n_buffered - n_flush;
        memmove(c->buf_keys,   c->buf_keys   + n_flush * stride, remaining * stride * sizeof(float));
        memmove(c->buf_values, c->buf_values + n_flush * stride, remaining * stride * sizeof(float));
        c->n_buffered = remaining;
    }
}

void t1b_kv_cache_prefill(t1b_kv_cache *c, const float *keys, const float *values, int seq_len) {
    int stride = hd_stride(c);
    int bs = c->config.buffer_size;

    if (seq_len <= bs) {
        // Everything fits in buffer
        memcpy(c->buf_keys,   keys,   seq_len * stride * sizeof(float));
        memcpy(c->buf_values, values, seq_len * stride * sizeof(float));
        c->n_buffered = seq_len;
        return;
    }

    // Compress older tokens, buffer recent ones
    int n_quant = seq_len - bs;
    grow_compressed(c, n_quant);

    for (int t = 0; t < n_quant; t++) {
        compress_token(c,
                       keys   + t * stride,
                       values + t * stride,
                       c->n_compressed + t);
    }
    c->n_compressed += n_quant;

    // Copy recent tokens to buffer
    memcpy(c->buf_keys,   keys   + n_quant * stride, bs * stride * sizeof(float));
    memcpy(c->buf_values, values + n_quant * stride, bs * stride * sizeof(float));
    c->n_buffered = bs;
}

// ── Attention computation ───────────────────────────────────────────

void t1b_kv_cache_attend(t1b_kv_cache *c,
                         const float *query,
                         float *output,
                         float scale) {
    int nh = c->config.n_heads;
    int hd = c->config.head_dim;
    int total = c->n_compressed + c->n_buffered;

    if (total == 0) {
        memset(output, 0, nh * hd * sizeof(float));
        return;
    }

    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)hd);
    }

    // Heap-allocate score array (may be large for long contexts)
    float *scores = (float *)malloc((size_t)total * sizeof(float));
    float *q_sketched = (float *)malloc((size_t)hd * sizeof(float));
    float *v_dequant = (float *)malloc((size_t)hd * sizeof(float));
    if (!scores || !q_sketched || !v_dequant) {
        free(scores); free(q_sketched); free(v_dequant);
        memset(output, 0, (size_t)nh * hd * sizeof(float));
        return;
    }

    // Process each head independently
    for (int h = 0; h < nh; h++) {
        const float *q = query + h * hd;
        float *out = output + h * hd;

        // Pre-sketch query for QJL scoring
        if (c->n_compressed > 0) {
            t1b_sketch_query(c->quantizers[h], q, q_sketched);
        }

        // Score compressed tokens
        for (int t = 0; t < c->n_compressed; t++) {
            int idx = t * nh + h;
            t1b_prod_quantized pq = {
                .mse_indices    = c->comp_keys[idx].mse_indices,
                .qjl_signs      = c->comp_keys[idx].qjl_signs,
                .residual_norm  = c->comp_keys[idx].residual_norm,
                .norm           = c->comp_keys[idx].norm,
                .mse_bits       = c->config.key_bits - 1,
                .d              = hd,
            };
            scores[t] = t1b_attention_score(c->quantizers[h], q, q_sketched, &pq) * scale;
        }

        // Score buffered tokens (full precision dot product)
        for (int t = 0; t < c->n_buffered; t++) {
            const float *k = c->buf_keys + t * nh * hd + h * hd;
            float dot = 0.0f;
            for (int i = 0; i < hd; i++) {
                dot += q[i] * k[i];
            }
            scores[c->n_compressed + t] = dot * scale;
        }

        // Online softmax (numerically stable)
        float max_score = -FLT_MAX;
        for (int t = 0; t < total; t++) {
            if (scores[t] > max_score) max_score = scores[t];
        }

        float sum_exp = 0.0f;
        for (int t = 0; t < total; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }

        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (int t = 0; t < total; t++) {
            scores[t] *= inv_sum;
        }

        // Weighted sum of values
        memset(out, 0, hd * sizeof(float));

        // Compressed values
        for (int t = 0; t < c->n_compressed; t++) {
            int idx = t * nh + h;
            t1b_value_quantized vq = {
                .data       = c->comp_values[idx].data,
                .scales     = c->comp_values[idx].scales,
                .zeros      = c->comp_values[idx].zeros,
                .bits       = c->config.value_bits,
                .d          = hd,
                .group_size = c->config.value_group_size,
                .n_groups   = c->val_n_groups,
            };
            t1b_dequantize_values(&vq, v_dequant);

            float w = scores[t];
            for (int i = 0; i < hd; i++) {
                out[i] += w * v_dequant[i];
            }
        }

        // Buffered values (full precision)
        for (int t = 0; t < c->n_buffered; t++) {
            const float *v = c->buf_values + t * nh * hd + h * hd;
            float w = scores[c->n_compressed + t];
            for (int i = 0; i < hd; i++) {
                out[i] += w * v[i];
            }
        }
    }

    free(scores);
    free(q_sketched);
    free(v_dequant);
}

// ── Memory stats ────────────────────────────────────────────────────

void t1b_kv_cache_memory_stats(const t1b_kv_cache *c, t1b_memory_stats *stats) {
    int nh = c->config.n_heads;
    int hd = c->config.head_dim;

    memset(stats, 0, sizeof(t1b_memory_stats));
    stats->n_compressed_tokens = c->n_compressed;
    stats->n_buffered_tokens = c->n_buffered;

    // Compressed keys: MSE indices + QJL signs + 2 floats (norms) per head per token
    size_t key_per_head = c->mse_packed_len + c->sign_packed_len + 2 * sizeof(float);
    stats->compressed_keys_bytes = (size_t)c->n_compressed * nh * key_per_head;

    // Compressed values: packed data + scales + zeros per head per token
    size_t val_per_head = c->val_packed_len + c->val_n_groups * 2 * sizeof(float);
    stats->compressed_values_bytes = (size_t)c->n_compressed * nh * val_per_head;

    // Buffer: full fp32
    stats->buffer_bytes = (size_t)c->n_buffered * nh * hd * sizeof(float) * 2; // keys + values

    // Overhead: rotation matrices (Pi + S per head, but shared in practice)
    stats->overhead_bytes = (size_t)nh * hd * hd * sizeof(float) * 2; // Pi + S per quantizer

    stats->total_bytes = stats->compressed_keys_bytes + stats->compressed_values_bytes
                       + stats->buffer_bytes + stats->overhead_bytes;

    // Compression ratio vs. fp16 for all tokens
    int total_tokens = c->n_compressed + c->n_buffered;
    if (total_tokens > 0) {
        size_t fp16_bytes = (size_t)total_tokens * nh * hd * 2 * 2; // keys+values in fp16
        stats->compression_ratio = (float)fp16_bytes / (float)(stats->total_bytes - stats->overhead_bytes);
    }
}
