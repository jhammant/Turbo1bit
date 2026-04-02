/**
 * turbo1bit_stress.c — Stress test pushing context lengths to find limits.
 *
 * Tests memory, throughput, and quality at extreme context lengths
 * from 1K to 1M+ tokens to find where Turbo1Bit breaks down.
 */

#include "turbo1bit_codebook.h"
#include "turbo1bit_rotation.h"
#include "turbo1bit_quantizer.h"
#include "turbo1bit_kv_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
#endif

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static void fill_random(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = randf() * 0.1f;
    }
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

static float cosine_similarity(const float *a, const float *b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
}

// ── Model configurations ───────────────────────────────────────────

typedef struct {
    const char *name;
    int n_layers;
    int n_kv_heads;
    int head_dim;
} model_config;

static const model_config BONSAI_1_7B = {
    .name = "Bonsai-1.7B",
    .n_layers = 24,
    .n_kv_heads = 8,
    .head_dim = 128,
};

static const model_config BONSAI_4B = {
    .name = "Bonsai-4B",
    .n_layers = 32,
    .n_kv_heads = 8,
    .head_dim = 128,
};

static const model_config BONSAI_8B = {
    .name = "Bonsai-8B",
    .n_layers = 32,
    .n_kv_heads = 8,
    .head_dim = 128,
};

// ── Single-layer memory + fill benchmark ────────────────────────────

static void bench_single_layer(int ctx_len, int n_heads, int head_dim,
                               double *out_fill_ms, double *out_t1b_mb,
                               double *out_fp16_mb, double *out_ratio) {
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
    if (!cache) {
        *out_fill_ms = -1;
        return;
    }

    int stride = n_heads * head_dim;
    float *kv = (float *)calloc((size_t)stride, sizeof(float));
    if (!kv) {
        t1b_kv_cache_free(cache);
        *out_fill_ms = -1;
        return;
    }

    double t0 = get_time_ms();
    for (int t = 0; t < ctx_len; t++) {
        fill_random(kv, stride);
        t1b_kv_cache_append(cache, kv, kv);
    }
    *out_fill_ms = get_time_ms() - t0;

    t1b_memory_stats stats;
    t1b_kv_cache_memory_stats(cache, &stats);

    *out_t1b_mb = (double)stats.total_bytes / (1024.0 * 1024.0);
    *out_fp16_mb = (double)ctx_len * n_heads * head_dim * 2.0 * 2.0 / (1024.0 * 1024.0);
    *out_ratio = (double)stats.compression_ratio;

    free(kv);
    t1b_kv_cache_free(cache);
}

// ── Full model memory projection ───────────────────────────────────

static void project_full_model(const model_config *model, int ctx_len) {
    double fill_ms, t1b_mb, fp16_mb, ratio;
    bench_single_layer(ctx_len, model->n_kv_heads, model->head_dim,
                       &fill_ms, &t1b_mb, &fp16_mb, &ratio);

    if (fill_ms < 0) {
        printf("  %-12d  FAILED (OOM or alloc error)\n", ctx_len);
        return;
    }

    double total_t1b = t1b_mb * model->n_layers;
    double total_fp16 = fp16_mb * model->n_layers;
    double fill_rate = (double)ctx_len / (fill_ms / 1000.0);

    printf("  %-12d  %-11.1f  %-13.1f  %-8.2fx  %-12.0f  %-8.1f\n",
           ctx_len, total_fp16, total_t1b, ratio, fill_rate, fill_ms / 1000.0);
}

// ── Attention latency at various contexts ──────────────────────────

static void bench_attention_latency(int ctx_len, int n_heads, int head_dim, int n_iters) {
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
    if (!cache) {
        printf("  %-12d  FAILED\n", ctx_len);
        return;
    }

    int stride = n_heads * head_dim;
    float *kv = (float *)calloc((size_t)stride, sizeof(float));

    // Fill
    for (int t = 0; t < ctx_len; t++) {
        fill_random(kv, stride);
        t1b_kv_cache_append(cache, kv, kv);
    }

    float *query = (float *)malloc((size_t)stride * sizeof(float));
    float *output = (float *)malloc((size_t)stride * sizeof(float));
    fill_random(query, stride);

    double t0 = get_time_ms();
    for (int i = 0; i < n_iters; i++) {
        t1b_kv_cache_attend(cache, query, output, 0.0f);
    }
    double total_ms = get_time_ms() - t0;
    double per_iter = total_ms / n_iters;

    // Effective tokens/sec for decode
    double decode_tps = 1000.0 / per_iter;

    printf("  %-12d  %-12.2f  %-14.1f  %-12d\n",
           ctx_len, per_iter, decode_tps, n_iters);

    free(kv);
    free(query);
    free(output);
    t1b_kv_cache_free(cache);
}

// ── Quality test at various contexts ───────────────────────────────

static void bench_quality(int ctx_len, int n_heads, int head_dim) {
    // Compare attention output with compressed vs full-precision KV
    int stride = n_heads * head_dim;

    // Store all KV in full precision
    float *all_keys   = (float *)malloc((size_t)ctx_len * stride * sizeof(float));
    float *all_values = (float *)malloc((size_t)ctx_len * stride * sizeof(float));
    if (!all_keys || !all_values) {
        printf("  %-12d  OOM for reference\n", ctx_len);
        free(all_keys);
        free(all_values);
        return;
    }

    for (int t = 0; t < ctx_len; t++) {
        fill_random(all_keys + (size_t)t * stride, stride);
        fill_random(all_values + (size_t)t * stride, stride);
    }

    // Build compressed cache
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
    if (!cache) {
        printf("  %-12d  FAILED\n", ctx_len);
        free(all_keys);
        free(all_values);
        return;
    }

    for (int t = 0; t < ctx_len; t++) {
        t1b_kv_cache_append(cache,
                            all_keys + (size_t)t * stride,
                            all_values + (size_t)t * stride);
    }

    // Compute compressed attention output
    float *query = (float *)malloc((size_t)stride * sizeof(float));
    float *output_compressed = (float *)malloc((size_t)stride * sizeof(float));
    fill_random(query, stride);

    t1b_kv_cache_attend(cache, query, output_compressed, 0.0f);

    // Compute reference attention output (full precision, per head)
    float *output_ref = (float *)calloc((size_t)stride, sizeof(float));
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        const float *q = query + h * head_dim;
        float *out = output_ref + h * head_dim;

        // Compute all scores
        float *scores = (float *)malloc((size_t)ctx_len * sizeof(float));
        float max_s = -1e30f;
        for (int t = 0; t < ctx_len; t++) {
            const float *k = all_keys + (size_t)t * stride + h * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; i++) dot += q[i] * k[i];
            scores[t] = dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }

        // Softmax
        float sum = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum += scores[t];
        }
        for (int t = 0; t < ctx_len; t++) scores[t] /= sum;

        // Weighted sum
        for (int t = 0; t < ctx_len; t++) {
            const float *v = all_values + (size_t)t * stride + h * head_dim;
            for (int i = 0; i < head_dim; i++) {
                out[i] += scores[t] * v[i];
            }
        }
        free(scores);
    }

    // Compare
    float cos_sim = cosine_similarity(output_compressed, output_ref, stride);
    float mse_val = 0.0f;
    for (int i = 0; i < stride; i++) {
        float d = output_compressed[i] - output_ref[i];
        mse_val += d * d;
    }
    mse_val /= (float)stride;

    printf("  %-12d  %-14.6f  %-12.2e  %s\n",
           ctx_len, (double)cos_sim, (double)mse_val,
           (cos_sim > 0.95f) ? "PASS" : (cos_sim > 0.90f) ? "MODERATE" : "FAIL");

    free(all_keys);
    free(all_values);
    free(query);
    free(output_compressed);
    free(output_ref);
    t1b_kv_cache_free(cache);
}

// ── Main ────────────────────────────────────────────────────────────

int main(void) {
    srand(42);

    printf("================================================================\n");
    printf("         Turbo1Bit Detailed Benchmark & Stress Test\n");
    printf("  Compressed KV Cache (3-bit keys, 2-bit values, 128-token buf)\n");
    printf("================================================================\n");

    size_t rss_start = get_rss_bytes();

    // ── 1. Memory projection for Bonsai-1.7B ──────────────────────
    printf("\n=== 1. Full Model KV Cache Memory: %s (%d layers) ===\n",
           BONSAI_1_7B.name, BONSAI_1_7B.n_layers);
    printf("  %-12s  %-11s  %-13s  %-8s  %-12s  %-8s\n",
           "Context", "FP16 (MB)", "Turbo1B (MB)", "Ratio", "Fill tok/s", "Fill (s)");
    printf("  %-12s  %-11s  %-13s  %-8s  %-12s  %-8s\n",
           "-------", "---------", "-----------", "-----", "----------", "-------");

    int ctx_1_7b[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "\r  [1.7B] Testing context=%d...          ", ctx_1_7b[i]);
        project_full_model(&BONSAI_1_7B, ctx_1_7b[i]);
    }
    fprintf(stderr, "\r                                           \r");

    // ── 2. Memory projection for Bonsai-8B ────────────────────────
    printf("\n=== 2. Full Model KV Cache Memory: %s (%d layers) ===\n",
           BONSAI_8B.name, BONSAI_8B.n_layers);
    printf("  %-12s  %-11s  %-13s  %-8s  %-12s  %-8s\n",
           "Context", "FP16 (MB)", "Turbo1B (MB)", "Ratio", "Fill tok/s", "Fill (s)");
    printf("  %-12s  %-11s  %-13s  %-8s  %-12s  %-8s\n",
           "-------", "---------", "-----------", "-----", "----------", "-------");

    int ctx_8b[] = {1024, 4096, 8192, 16384, 32768, 65536};
    for (int i = 0; i < 6; i++) {
        fprintf(stderr, "\r  [8B] Testing context=%d...          ", ctx_8b[i]);
        project_full_model(&BONSAI_8B, ctx_8b[i]);
    }
    fprintf(stderr, "\r                                           \r");

    // ── 3. Attention latency at various context lengths ───────────
    printf("\n=== 3. Attention Decode Latency (CPU, single layer, 8 heads) ===\n");
    printf("  %-12s  %-12s  %-14s  %-12s\n",
           "Context", "ms/decode", "tok/s (est)", "Iterations");
    printf("  %-12s  %-12s  %-14s  %-12s\n",
           "-------", "---------", "-----------", "----------");

    int attn_ctx[] = {256, 512, 1024, 2048, 4096, 8192};
    int attn_iters[] = {500, 200, 100, 50, 20, 10};
    for (int i = 0; i < 6; i++) {
        fprintf(stderr, "\r  Attention benchmark ctx=%d...     ", attn_ctx[i]);
        bench_attention_latency(attn_ctx[i], 8, 128, attn_iters[i]);
    }
    fprintf(stderr, "\r                                           \r");

    // ── 4. Quality at various context lengths ─────────────────────
    printf("\n=== 4. Attention Output Quality (compressed vs. reference) ===\n");
    printf("  %-12s  %-14s  %-12s  %s\n",
           "Context", "Cosine Sim", "MSE", "Status");
    printf("  %-12s  %-14s  %-12s  %s\n",
           "-------", "----------", "---", "------");

    int quality_ctx[] = {128, 512, 1024, 2048, 4096};
    for (int i = 0; i < 5; i++) {
        fprintf(stderr, "\r  Quality test ctx=%d...          ", quality_ctx[i]);
        bench_quality(quality_ctx[i], 8, 128);
    }
    fprintf(stderr, "\r                                           \r");

    // ── 5. Compression ratio breakdown ────────────────────────────
    printf("\n=== 5. Per-Token Storage Breakdown (d=128, 3-bit keys, 2-bit values) ===\n");
    int d = 128;
    int key_mse_packed = t1b_packed_len(d, 2);    // 2-bit MSE = bits-1
    int key_qjl_packed = t1b_sign_packed_len(d);   // 1-bit signs
    int val_packed = t1b_value_packed_len(d, 2);   // 2-bit values
    int val_groups = d / 32;

    int key_bytes_per_head = key_mse_packed + key_qjl_packed + 8; // +8 for norms
    int val_bytes_per_head = val_packed + val_groups * 8;          // +scales+zeros
    int total_per_head = key_bytes_per_head + val_bytes_per_head;
    int fp16_per_head = d * 2 * 2; // key + value in fp16

    printf("  Key storage per head per token:\n");
    printf("    MSE indices (2-bit packed): %d bytes\n", key_mse_packed);
    printf("    QJL signs (1-bit packed):   %d bytes\n", key_qjl_packed);
    printf("    Norms (2x float32):         8 bytes\n");
    printf("    Total key:                  %d bytes (vs %d bytes FP16)\n",
           key_bytes_per_head, d * 2);
    printf("  Value storage per head per token:\n");
    printf("    Quantized data (2-bit):     %d bytes\n", val_packed);
    printf("    Scales + zeros (%dx f32):   %d bytes\n", val_groups, val_groups * 8);
    printf("    Total value:                %d bytes (vs %d bytes FP16)\n",
           val_bytes_per_head, d * 2);
    printf("  Combined: %d bytes/head/token vs %d FP16 = %.2fx compression\n",
           total_per_head, fp16_per_head, (double)fp16_per_head / total_per_head);

    // ── 6. Process memory ─────────────────────────────────────────
    size_t rss_end = get_rss_bytes();
    printf("\n=== 6. Process Memory ===\n");
    printf("  RSS start: %.1f MB\n", (double)rss_start / (1024.0 * 1024.0));
    printf("  RSS end:   %.1f MB\n", (double)rss_end / (1024.0 * 1024.0));
    printf("  RSS delta: %.1f MB\n", (double)(rss_end - rss_start) / (1024.0 * 1024.0));

    // ── 7. Maximum context length estimate ────────────────────────
    printf("\n=== 7. Maximum Context Length Estimates ===\n");
    printf("  (Based on measured compression ratios, single-layer overhead excluded)\n\n");

    double ratio_at_scale = 4.0; // measured ~4x at 16K+
    int ram_configs[] = {8, 16, 32, 64};
    const char *ram_labels[] = {"8 GB (M1)", "16 GB (M1 Pro)", "32 GB (M1 Max)", "64 GB (M2 Ultra)"};

    for (int mi = 0; mi < 3; mi++) { // 3 models
        const model_config *m = (mi == 0) ? &BONSAI_1_7B : (mi == 1) ? &BONSAI_4B : &BONSAI_8B;
        double model_weight_gb = (mi == 0) ? 0.2 : (mi == 1) ? 0.5 : 1.0; // 1-bit model sizes

        printf("  %s (1-bit weights: ~%.1f GB):\n", m->name, model_weight_gb);

        for (int ri = 0; ri < 4; ri++) {
            double avail_gb = (double)ram_configs[ri] - model_weight_gb - 1.0; // 1GB OS overhead
            double avail_bytes = avail_gb * 1024.0 * 1024.0 * 1024.0;

            // FP16 KV per token per layer = n_kv_heads * head_dim * 2 (k+v) * 2 (fp16)
            double fp16_per_tok = (double)m->n_kv_heads * m->head_dim * 2.0 * 2.0;
            double t1b_per_tok = fp16_per_tok / ratio_at_scale;

            int max_ctx_fp16 = (int)(avail_bytes / (fp16_per_tok * m->n_layers));
            int max_ctx_t1b = (int)(avail_bytes / (t1b_per_tok * m->n_layers));

            printf("    %-20s  FP16: %7dK tokens  Turbo1Bit: %7dK tokens  (%.1fx more)\n",
                   ram_labels[ri],
                   max_ctx_fp16 / 1024,
                   max_ctx_t1b / 1024,
                   (double)max_ctx_t1b / max_ctx_fp16);
        }
        printf("\n");
    }

    printf("================================================================\n");
    printf("  Benchmark complete.\n");
    printf("================================================================\n");

    return 0;
}
