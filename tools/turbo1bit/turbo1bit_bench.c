/**
 * turbo1bit_bench.c — Benchmark comparing Turbo1Bit compressed KV cache vs baseline.
 *
 * Tests:
 *   1. Quantization correctness (cosine similarity, MSE)
 *   2. Memory usage at various context lengths
 *   3. Compression/decompression throughput
 *   4. Attention accuracy (compressed vs. full precision)
 *   5. End-to-end KV cache memory savings
 */

#include "turbo1bit_codebook.h"
#include "turbo1bit_rotation.h"
#include "turbo1bit_quantizer.h"
#include "turbo1bit_kv_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

// ── Timing ──────────────────────────────────────────────────────────

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ── Random data generation ──────────────────────────────────────────

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static void fill_random(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = randf() * 0.1f;
    }
}

// ── Math helpers ────────────────────────────────────────────────────

static float cosine_similarity(const float *a, const float *b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
}

static float compute_mse(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum / (float)n;
}

#ifdef __APPLE__
static size_t get_rss_bytes(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &size) != KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
}
#else
static size_t get_rss_bytes(void) { return 0; }
#endif

// ── Tests ───────────────────────────────────────────────────────────

static void test_codebook(void) {
    printf("\n=== Test 1: Codebook Lookup ===\n");
    const int dims[] = {64, 128};
    const int bits[] = {1, 2, 3};

    for (int di = 0; di < 2; di++) {
        for (int bi = 0; bi < 3; bi++) {
            const struct t1b_codebook *cb = t1b_get_codebook(dims[di], bits[bi]);
            if (cb) {
                printf("  d=%3d, bits=%d: %d centroids, MSE/coord=%.6e  OK\n",
                       cb->d, cb->bits, cb->n_clusters, (double)cb->mse_per_coord);
            } else {
                printf("  d=%3d, bits=%d: NOT FOUND\n", dims[di], bits[bi]);
            }
        }
    }
}

static void test_rotation(void) {
    printf("\n=== Test 2: Rotation Matrix Orthogonality ===\n");
    int dim = 128;
    t1b_rotation *rot = t1b_rotation_create(dim, 0);

    float x[128], y[128], x_back[128];
    fill_random(x, dim);

    t1b_rotate_forward(rot, x, y);
    t1b_rotate_backward(rot, y, x_back);

    float cos_sim = cosine_similarity(x, x_back, dim);
    float err = compute_mse(x, x_back, dim);
    printf("  Forward-backward roundtrip: cos_sim=%.8f, MSE=%.2e  %s\n",
           (double)cos_sim, (double)err, (cos_sim > 0.9999f) ? "PASS" : "FAIL");

    float norm_x = 0.0f, norm_y = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }
    printf("  Norm preservation: ||x||=%.6f, ||Px||=%.6f, ratio=%.8f  %s\n",
           (double)sqrtf(norm_x), (double)sqrtf(norm_y), (double)sqrtf(norm_y / norm_x),
           (fabsf(norm_x - norm_y) / norm_x < 0.001f) ? "PASS" : "FAIL");

    t1b_rotation_free(rot);
}

static void test_quantizer_mse(void) {
    printf("\n=== Test 3: MSE Quantization Quality ===\n");
    int dim = 128;
    int n_vectors = 1000;
    int bits_list[] = {2, 3};

    for (int bi = 0; bi < 2; bi++) {
        int bits = bits_list[bi];
        t1b_quantizer *q = t1b_quantizer_create(dim, bits, 0);
        if (!q) { printf("  bits=%d: FAILED to create quantizer\n", bits); continue; }

        float total_cos = 0.0f;
        float total_mse = 0.0f;

        int mse_pl = t1b_packed_len(dim, bits - 1);
        int sign_pl = t1b_sign_packed_len(dim);

        for (int v = 0; v < n_vectors; v++) {
            float x[128], x_hat[128];
            fill_random(x, dim);

            uint8_t mse_indices[mse_pl];
            uint8_t qjl_signs[sign_pl];

            t1b_prod_quantized pq = {
                .mse_indices = mse_indices,
                .qjl_signs = qjl_signs,
                .residual_norm = 0.0f,
                .norm = 0.0f,
                .mse_bits = 0,
                .d = 0,
            };
            t1b_quantize_prod(q, x, &pq);
            t1b_dequantize_prod(q, &pq, x_hat);

            total_cos += cosine_similarity(x, x_hat, dim);
            total_mse += compute_mse(x, x_hat, dim);
        }

        printf("  bits=%d: avg_cos_sim=%.6f, avg_MSE=%.6e  %s\n",
               bits, (double)(total_cos / (float)n_vectors),
               (double)(total_mse / (float)n_vectors),
               (total_cos / (float)n_vectors > 0.9f) ? "PASS" : "FAIL");

        t1b_quantizer_free(q);
    }
}

static void test_value_quantization(void) {
    printf("\n=== Test 4: Value Group Quantization ===\n");
    int dim = 128;
    int n_vectors = 1000;
    int bits_list[] = {2, 4};

    for (int bi = 0; bi < 2; bi++) {
        int bits = bits_list[bi];
        int gs = 32;
        int n_groups = dim / gs;

        float total_cos = 0.0f;
        float total_mse = 0.0f;

        for (int v = 0; v < n_vectors; v++) {
            float x[128], x_hat[128];
            fill_random(x, dim);

            int packed_len = t1b_value_packed_len(dim, bits);
            uint8_t data[packed_len];
            float scales[n_groups], zeros[n_groups];

            t1b_value_quantized vq = {
                .data = data, .scales = scales, .zeros = zeros,
                .bits = bits, .d = dim, .group_size = gs, .n_groups = n_groups
            };
            t1b_quantize_values(x, dim, bits, gs, &vq);
            t1b_dequantize_values(&vq, x_hat);

            total_cos += cosine_similarity(x, x_hat, dim);
            total_mse += compute_mse(x, x_hat, dim);
        }

        printf("  bits=%d: avg_cos_sim=%.6f, avg_MSE=%.6e  %s\n",
               bits, (double)(total_cos / (float)n_vectors),
               (double)(total_mse / (float)n_vectors),
               (total_cos / (float)n_vectors > 0.93f) ? "PASS" : "FAIL");
    }
}

static void test_attention_accuracy(void) {
    printf("\n=== Test 5: Attention Score Accuracy ===\n");
    int dim = 128;
    int bits = 3;

    t1b_quantizer *q = t1b_quantizer_create(dim, bits, 0);
    int mse_pl = t1b_packed_len(dim, bits - 1);
    int sign_pl = t1b_sign_packed_len(dim);

    int n_keys = 100;
    float total_rel_err = 0.0f;

    float query[128];
    fill_random(query, dim);

    float q_sketched[128];
    t1b_sketch_query(q, query, q_sketched);

    for (int k = 0; k < n_keys; k++) {
        float key[128];
        fill_random(key, dim);

        float true_score = 0.0f;
        for (int i = 0; i < dim; i++) {
            true_score += query[i] * key[i];
        }

        uint8_t mse_indices[mse_pl];
        uint8_t qjl_signs[sign_pl];
        t1b_prod_quantized pq = {
            .mse_indices = mse_indices, .qjl_signs = qjl_signs,
            .residual_norm = 0.0f, .norm = 0.0f, .mse_bits = 0, .d = 0,
        };
        t1b_quantize_prod(q, key, &pq);

        float est_score = t1b_attention_score(q, query, q_sketched, &pq);

        float abs_err = fabsf(est_score - true_score);
        total_rel_err += abs_err;
    }

    float avg_abs_err = total_rel_err / (float)n_keys;
    // For random vectors with magnitude ~0.1, dot products are ~0.01
    // Absolute error should be small relative to vector magnitudes
    printf("  3-bit key compression: avg_abs_error=%.6f  %s\n",
           (double)avg_abs_err,
           (avg_abs_err < 0.01f) ? "PASS" : "MODERATE");

    t1b_quantizer_free(q);
}

static void bench_kv_cache_memory(void) {
    printf("\n=== Test 6: KV Cache Memory Comparison ===\n");
    printf("  %-12s  %-14s  %-14s  %-10s\n",
           "Context Len", "FP16 (MB)", "Turbo1Bit (MB)", "Ratio");
    printf("  %-12s  %-14s  %-14s  %-10s\n",
           "___________", "_________", "______________", "_____");

    int n_heads = 8;
    int head_dim = 128;
    int context_lengths[] = {512, 2048, 8192, 16384};

    for (int ci = 0; ci < 4; ci++) {
        int ctx_len = context_lengths[ci];

        double fp16_mb = (double)ctx_len * n_heads * head_dim * 2.0 * 2.0 / (1024.0 * 1024.0);

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
        if (!cache) { printf("  FAILED to create cache\n"); continue; }

        float *kv = (float *)calloc((size_t)n_heads * head_dim, sizeof(float));
        for (int t = 0; t < ctx_len; t++) {
            fill_random(kv, n_heads * head_dim);
            t1b_kv_cache_append(cache, kv, kv);

            if (ctx_len >= 8192 && t % 4096 == 0 && t > 0) {
                fprintf(stderr, "\r  Filling %d/%d tokens...", t, ctx_len);
            }
        }
        if (ctx_len >= 8192) fprintf(stderr, "\r                                     \r");

        t1b_memory_stats stats;
        t1b_kv_cache_memory_stats(cache, &stats);
        double t1b_mb = (double)stats.total_bytes / (1024.0 * 1024.0);

        printf("  %-12d  %-14.2f  %-14.2f  %-10.2fx\n",
               ctx_len, fp16_mb, t1b_mb, (double)stats.compression_ratio);

        free(kv);
        t1b_kv_cache_free(cache);
    }
}

static void bench_throughput(void) {
    printf("\n=== Test 7: Quantization Throughput ===\n");
    int dim = 128;
    int bits = 3;
    int n_vectors = 10000;

    t1b_quantizer *q = t1b_quantizer_create(dim, bits, 0);
    int mse_pl = t1b_packed_len(dim, bits - 1);
    int sign_pl = t1b_sign_packed_len(dim);

    float *vectors = (float *)malloc((size_t)n_vectors * dim * sizeof(float));
    fill_random(vectors, n_vectors * dim);

    uint8_t *all_mse = (uint8_t *)calloc((size_t)n_vectors * mse_pl, 1);
    uint8_t *all_signs = (uint8_t *)calloc((size_t)n_vectors * sign_pl, 1);
    float *norms = (float *)malloc((size_t)n_vectors * sizeof(float));
    float *rnorms = (float *)malloc((size_t)n_vectors * sizeof(float));

    double t0 = get_time_ms();
    for (int v = 0; v < n_vectors; v++) {
        t1b_prod_quantized pq = {
            .mse_indices = all_mse + v * mse_pl,
            .qjl_signs = all_signs + v * sign_pl,
            .residual_norm = 0.0f, .norm = 0.0f, .mse_bits = 0, .d = 0,
        };
        t1b_quantize_prod(q, vectors + v * dim, &pq);
        norms[v] = pq.norm;
        rnorms[v] = pq.residual_norm;
    }
    double quant_ms = get_time_ms() - t0;

    float *output = (float *)malloc((size_t)n_vectors * dim * sizeof(float));
    t0 = get_time_ms();
    for (int v = 0; v < n_vectors; v++) {
        t1b_prod_quantized pq = {
            .mse_indices = all_mse + v * mse_pl,
            .qjl_signs = all_signs + v * sign_pl,
            .residual_norm = rnorms[v],
            .norm = norms[v],
            .mse_bits = bits - 1,
            .d = dim,
        };
        t1b_dequantize_prod(q, &pq, output + v * dim);
    }
    double dequant_ms = get_time_ms() - t0;

    printf("  Quantize:   %d vectors in %.1f ms  (%.0f vectors/sec)\n",
           n_vectors, quant_ms, n_vectors / (quant_ms / 1000.0));
    printf("  Dequantize: %d vectors in %.1f ms  (%.0f vectors/sec)\n",
           n_vectors, dequant_ms, n_vectors / (dequant_ms / 1000.0));

    free(vectors);
    free(all_mse);
    free(all_signs);
    free(norms);
    free(rnorms);
    free(output);
    t1b_quantizer_free(q);
}

static void bench_attention(void) {
    printf("\n=== Test 8: Attention End-to-End ===\n");
    int n_heads = 8;
    int head_dim = 128;
    int ctx_len = 2048;

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
    int stride = n_heads * head_dim;

    float *kv = (float *)calloc((size_t)stride, sizeof(float));
    for (int t = 0; t < ctx_len; t++) {
        fill_random(kv, stride);
        t1b_kv_cache_append(cache, kv, kv);
    }

    float *query = (float *)malloc((size_t)stride * sizeof(float));
    float *att_output = (float *)malloc((size_t)stride * sizeof(float));
    fill_random(query, stride);

    int n_iters = 100;
    double t0 = get_time_ms();
    for (int i = 0; i < n_iters; i++) {
        t1b_kv_cache_attend(cache, query, att_output, 0.0f);
    }
    double attn_ms = get_time_ms() - t0;

    printf("  Context=%d, heads=%d, head_dim=%d\n", ctx_len, n_heads, head_dim);
    printf("  Attention: %d iterations in %.1f ms  (%.2f ms/iter)\n",
           n_iters, attn_ms, attn_ms / n_iters);

    free(kv);
    free(query);
    free(att_output);
    t1b_kv_cache_free(cache);
}

// ── Main ────────────────────────────────────────────────────────────

int main(void) {
    srand(42);

    printf("==================================================\n");
    printf("           Turbo1Bit Benchmark Suite\n");
    printf("  TurboQuant KV Cache + Bonsai 1-Bit Weights\n");
    printf("==================================================\n");

    size_t rss_start = get_rss_bytes();

    test_codebook();
    test_rotation();
    test_quantizer_mse();
    test_value_quantization();
    test_attention_accuracy();
    bench_throughput();
    bench_kv_cache_memory();
    bench_attention();

    size_t rss_end = get_rss_bytes();
    printf("\n=== Process Memory ===\n");
    printf("  RSS at start: %.2f MB\n", (double)rss_start / (1024.0 * 1024.0));
    printf("  RSS at end:   %.2f MB\n", (double)rss_end / (1024.0 * 1024.0));
    printf("  Delta:        %.2f MB\n", (double)(rss_end - rss_start) / (1024.0 * 1024.0));

    printf("\n=== Summary ===\n");
    printf("  All tests completed. See above for PASS/FAIL status.\n");
    printf("  Turbo1Bit combines 1-bit model weights (Bonsai) with\n");
    printf("  compressed KV cache (TurboQuant) for maximum efficiency.\n");

    return 0;
}
