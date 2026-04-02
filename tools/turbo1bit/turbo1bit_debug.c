/**
 * turbo1bit_debug.c — Debug test to identify quality issue in attention output.
 */

#include "../../src/turbo1bit_codebook.h"
#include "../../src/turbo1bit_rotation.h"
#include "../../src/turbo1bit_quantizer.h"
#include "../../src/turbo1bit_kv_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static void fill_random(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = randf() * 0.1f;
    }
}

static float cosine_similarity(const float *a, const float *b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
}

int main(void) {
    srand(42);

    int n_heads = 1;  // Single head for clarity
    int head_dim = 128;
    int stride = n_heads * head_dim;

    printf("=== Debug: Single-head attention quality ===\n\n");

    // Test with 256 tokens (128 compressed + 128 buffered)
    int ctx_len = 256;

    // Store reference keys and values
    float *ref_keys   = (float *)malloc((size_t)ctx_len * stride * sizeof(float));
    float *ref_values = (float *)malloc((size_t)ctx_len * stride * sizeof(float));

    // Use separate PRNG seeds for keys and values to match cache behavior
    srand(42);
    for (int t = 0; t < ctx_len; t++) {
        fill_random(ref_keys   + (size_t)t * stride, stride);
        fill_random(ref_values + (size_t)t * stride, stride);
    }

    // Build compressed cache with same data
    srand(42);
    t1b_kv_cache_config config = {
        .head_dim = head_dim,
        .n_heads = n_heads,
        .key_bits = 3,
        .value_bits = 2,
        .value_group_size = 32,
        .buffer_size = 128,
        .layer_idx = 0,
        .max_seq_len = ctx_len,
    };

    t1b_kv_cache *cache = t1b_kv_cache_create(&config);
    float *token_k = (float *)malloc(stride * sizeof(float));
    float *token_v = (float *)malloc(stride * sizeof(float));

    for (int t = 0; t < ctx_len; t++) {
        fill_random(token_k, stride);
        fill_random(token_v, stride);
        t1b_kv_cache_append(cache, token_k, token_v);
    }

    t1b_memory_stats stats;
    t1b_kv_cache_memory_stats(cache, &stats);
    printf("Cache state: %d compressed, %d buffered, total=%d\n",
           stats.n_compressed_tokens, stats.n_buffered_tokens,
           t1b_kv_cache_seq_len(cache));

    // Query
    srand(12345);
    float *query = (float *)malloc(stride * sizeof(float));
    fill_random(query, stride);

    // Compressed attention output
    float *output_t1b = (float *)malloc(stride * sizeof(float));
    t1b_kv_cache_attend(cache, query, output_t1b, 0.0f);

    // Reference attention output
    float *output_ref = (float *)calloc(stride, sizeof(float));
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        const float *q = query + h * head_dim;
        float *out = output_ref + h * head_dim;

        float *scores = (float *)malloc((size_t)ctx_len * sizeof(float));
        float max_s = -1e30f;

        for (int t = 0; t < ctx_len; t++) {
            const float *k = ref_keys + (size_t)t * stride + h * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; i++) dot += q[i] * k[i];
            scores[t] = dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }

        // Print score statistics
        float min_s = scores[0], sum_s = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            if (scores[t] < min_s) min_s = scores[t];
            sum_s += scores[t];
        }
        printf("\nRef scores: min=%.6f, max=%.6f, mean=%.6f\n",
               (double)min_s, (double)max_s, (double)(sum_s / ctx_len));

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }
        for (int t = 0; t < ctx_len; t++) scores[t] /= sum_exp;

        // Print top attention weights
        printf("Top 5 attention weights:\n");
        for (int iter = 0; iter < 5; iter++) {
            int best = 0;
            for (int t = 1; t < ctx_len; t++) {
                if (scores[t] > scores[best]) best = t;
            }
            printf("  Token %d: weight=%.6f\n", best, (double)scores[best]);
            scores[best] = -1.0f; // mark used
        }

        // Re-softmax for actual computation
        sum_exp = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            const float *k = ref_keys + (size_t)t * stride + h * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; i++) dot += q[i] * k[i];
            scores[t] = dot * scale;
        }
        max_s = -1e30f;
        for (int t = 0; t < ctx_len; t++) if (scores[t] > max_s) max_s = scores[t];
        for (int t = 0; t < ctx_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }
        for (int t = 0; t < ctx_len; t++) scores[t] /= sum_exp;

        for (int t = 0; t < ctx_len; t++) {
            const float *v = ref_values + (size_t)t * stride + h * head_dim;
            for (int i = 0; i < head_dim; i++) {
                out[i] += scores[t] * v[i];
            }
        }
        free(scores);
    }

    float cos_output = cosine_similarity(output_t1b, output_ref, stride);
    printf("\nAttention output cosine similarity: %.6f\n", (double)cos_output);

    // Print first few values
    printf("\nFirst 8 output values:\n");
    printf("  T1B: ");
    for (int i = 0; i < 8; i++) printf("%.6f ", (double)output_t1b[i]);
    printf("\n  Ref: ");
    for (int i = 0; i < 8; i++) printf("%.6f ", (double)output_ref[i]);
    printf("\n");

    // Check output magnitudes
    float norm_t1b = 0.0f, norm_ref = 0.0f;
    for (int i = 0; i < stride; i++) {
        norm_t1b += output_t1b[i] * output_t1b[i];
        norm_ref += output_ref[i] * output_ref[i];
    }
    printf("\nOutput norms: T1B=%.6f, Ref=%.6f\n",
           (double)sqrtf(norm_t1b), (double)sqrtf(norm_ref));

    free(ref_keys); free(ref_values);
    free(query); free(output_t1b); free(output_ref);
    free(token_k); free(token_v);
    t1b_kv_cache_free(cache);

    // ── Test 2: Realistic attention pattern ────────────────────────
    printf("\n\n=== Debug: Realistic attention pattern (concentrated) ===\n");

    srand(42);
    ctx_len = 512;
    ref_keys   = (float *)malloc((size_t)ctx_len * stride * sizeof(float));
    ref_values = (float *)malloc((size_t)ctx_len * stride * sizeof(float));

    // Create keys where a few "important" tokens align strongly with query
    query = (float *)malloc(stride * sizeof(float));
    fill_random(query, stride);

    // Make query have a stronger signal
    for (int i = 0; i < stride; i++) query[i] *= 5.0f;

    srand(42);
    for (int t = 0; t < ctx_len; t++) {
        fill_random(ref_keys + (size_t)t * stride, stride);
        fill_random(ref_values + (size_t)t * stride, stride);
    }

    // Make a few tokens match the query strongly (simulate "important" tokens)
    int important[] = {10, 50, 100, 200, 400};
    for (int idx = 0; idx < 5; idx++) {
        int t = important[idx];
        for (int i = 0; i < stride; i++) {
            ref_keys[(size_t)t * stride + i] = query[i] * (0.8f + randf() * 0.1f);
        }
    }

    // Build cache
    srand(42);
    config.max_seq_len = ctx_len;
    cache = t1b_kv_cache_create(&config);
    for (int t = 0; t < ctx_len; t++) {
        t1b_kv_cache_append(cache, ref_keys + (size_t)t * stride,
                            ref_values + (size_t)t * stride);
    }

    output_t1b = (float *)malloc(stride * sizeof(float));
    t1b_kv_cache_attend(cache, query, output_t1b, 0.0f);

    // Reference
    output_ref = (float *)calloc(stride, sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        const float *q2 = query + h * head_dim;
        float *out = output_ref + h * head_dim;

        float *scores2 = (float *)malloc((size_t)ctx_len * sizeof(float));
        float max_s2 = -1e30f;
        for (int t = 0; t < ctx_len; t++) {
            const float *k = ref_keys + (size_t)t * stride + h * head_dim;
            float dot2 = 0.0f;
            for (int i = 0; i < head_dim; i++) dot2 += q2[i] * k[i];
            scores2[t] = dot2 * scale;
            if (scores2[t] > max_s2) max_s2 = scores2[t];
        }
        float sum_exp2 = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            scores2[t] = expf(scores2[t] - max_s2);
            sum_exp2 += scores2[t];
        }
        for (int t = 0; t < ctx_len; t++) scores2[t] /= sum_exp2;

        for (int t = 0; t < ctx_len; t++) {
            const float *v = ref_values + (size_t)t * stride + h * head_dim;
            for (int i = 0; i < head_dim; i++) {
                out[i] += scores2[t] * v[i];
            }
        }
        free(scores2);
    }

    cos_output = cosine_similarity(output_t1b, output_ref, stride);
    printf("Attention output cosine similarity: %.6f  %s\n",
           (double)cos_output,
           (cos_output > 0.95f) ? "PASS" : (cos_output > 0.90f) ? "MODERATE" : "FAIL");

    norm_t1b = 0.0f; norm_ref = 0.0f;
    for (int i = 0; i < stride; i++) {
        norm_t1b += output_t1b[i] * output_t1b[i];
        norm_ref += output_ref[i] * output_ref[i];
    }
    printf("Output norms: T1B=%.6f, Ref=%.6f\n",
           (double)sqrtf(norm_t1b), (double)sqrtf(norm_ref));

    free(ref_keys); free(ref_values);
    free(query); free(output_t1b); free(output_ref);
    t1b_kv_cache_free(cache);

    return 0;
}
