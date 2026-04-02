/**
 * turbo1bit_kv_cache.h — Compressed KV cache combining TurboQuant key compression
 * with group-quantized values and a full-precision recent-token buffer.
 *
 * This provides the main integration point for llama.cpp: a cache that stores
 * older tokens in compressed form while keeping recent tokens at full precision.
 */

#ifndef TURBO1BIT_KV_CACHE_H
#define TURBO1BIT_KV_CACHE_H

#include "turbo1bit_quantizer.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Compressed KV cache for one attention layer ────────────────────

typedef struct t1b_kv_cache t1b_kv_cache;

// Configuration
typedef struct {
    int head_dim;           // embedding dimension per head (e.g., 128)
    int n_heads;            // number of KV heads
    int key_bits;           // total bits for key compression (default: 3)
    int value_bits;         // bits for value group quantization (default: 2)
    int value_group_size;   // group size for value quantization (default: 32)
    int buffer_size;        // number of recent tokens kept uncompressed (default: 128)
    int layer_idx;          // layer index for deterministic seeding
    int max_seq_len;        // maximum sequence length (for pre-allocation)
} t1b_kv_cache_config;

// Create compressed KV cache for one layer.
t1b_kv_cache * t1b_kv_cache_create(const t1b_kv_cache_config *config);
void           t1b_kv_cache_free(t1b_kv_cache *cache);

// Reset cache (clear all stored tokens).
void t1b_kv_cache_clear(t1b_kv_cache *cache);

// Get current sequence length (compressed + buffered tokens).
int t1b_kv_cache_seq_len(const t1b_kv_cache *cache);

// ── Storage ────────────────────────────────────────────────────────

// Store a token's KV pair. Handles buffer management and auto-compression.
// key and value are float arrays of length (n_heads * head_dim).
// Position is the sequence position of this token.
void t1b_kv_cache_append(t1b_kv_cache *cache,
                         const float *key,   // [n_heads, head_dim]
                         const float *value); // [n_heads, head_dim]

// Prefill: store multiple tokens at once (more efficient than repeated append).
void t1b_kv_cache_prefill(t1b_kv_cache *cache,
                          const float *keys,   // [seq_len, n_heads, head_dim]
                          const float *values,  // [seq_len, n_heads, head_dim]
                          int seq_len);

// ── Attention ──────────────────────────────────────────────────────

// Compute attention output for a single query token across all cached KV.
// query: [n_heads, head_dim] — one query per head
// output: [n_heads, head_dim] — attention output per head
// scale: attention scale factor (typically 1/sqrt(head_dim)), 0 for default
void t1b_kv_cache_attend(t1b_kv_cache *cache,
                         const float *query,   // [n_heads, head_dim]
                         float *output,        // [n_heads, head_dim]
                         float scale);

// ── Memory reporting ───────────────────────────────────────────────

typedef struct {
    size_t compressed_keys_bytes;
    size_t compressed_values_bytes;
    size_t buffer_bytes;
    size_t overhead_bytes;      // rotation matrices, codebooks, etc.
    size_t total_bytes;
    int    n_compressed_tokens;
    int    n_buffered_tokens;
    float  compression_ratio;   // vs. full fp16 storage
} t1b_memory_stats;

void t1b_kv_cache_memory_stats(const t1b_kv_cache *cache, t1b_memory_stats *stats);

#ifdef __cplusplus
}
#endif

#endif // TURBO1BIT_KV_CACHE_H
