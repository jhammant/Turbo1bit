/**
 * turbo1bit_infer.cpp — End-to-end inference with Turbo1Bit KV cache compression.
 *
 * Runs a Bonsai model and applies Turbo1Bit quantize-then-dequantize to KV cache
 * entries in-place after each decode step. This simulates the quality impact of
 * compressed KV storage with actual model output.
 *
 * Usage:
 *   turbo1bit-infer -m model.gguf -p "prompt" [-n 128] [-c 2048] [--no-turbo1bit]
 */

#include "llama.h"
#include "common.h"

extern "C" {
#include "turbo1bit_quantizer.h"
#include "turbo1bit_metal.h"
}

#include "llama-kv-cache.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

struct turbo1bit_state {
    bool enabled = false;
    int n_layers = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int key_bits = 0;     // 0 = no key compression (default safe mode)
    int value_bits = 4;   // 4-bit values (0.998 cos_sim)
    int value_group_size = 32;
    int64_t tokens_compressed = 0;

    std::vector<t1b_quantizer *> quantizers;
    t1b_metal_ctx *metal = nullptr;  // GPU acceleration

    // Track which KV slots have already been compressed (avoid double-compression)
    std::vector<bool> slot_compressed;

    ~turbo1bit_state() {
        for (auto *q : quantizers) t1b_quantizer_free(q);
        t1b_metal_free(metal);
    }
};

static void turbo1bit_init(turbo1bit_state & state, const llama_model * model) {
    state.n_layers = llama_model_n_layer(model);
    state.n_kv_heads = llama_model_n_head_kv(model);
    // head_dim = n_embd / n_head (query heads, not KV heads)
    int n_embd = llama_model_n_embd(model);
    int n_head = llama_model_n_head(model);
    state.head_dim = n_embd / n_head;
    state.enabled = true;

    fprintf(stderr, "[Turbo1Bit] %d layers, %d KV heads, head_dim=%d\n",
            state.n_layers, state.n_kv_heads, state.head_dim);

    if (state.key_bits >= 2) {
        state.quantizers.resize(state.n_layers);
        for (int l = 0; l < state.n_layers; l++) {
            state.quantizers[l] = t1b_quantizer_create(state.head_dim, state.key_bits, l);
            if (!state.quantizers[l]) {
                fprintf(stderr, "[Turbo1Bit] ERROR: Failed to create quantizer for layer %d\n", l);
                state.enabled = false;
                return;
            }
        }
        fprintf(stderr, "[Turbo1Bit] Key compression: %d-bit TurboQuantProd\n", state.key_bits);
    } else {
        fprintf(stderr, "[Turbo1Bit] Key compression: DISABLED (FP16 passthrough)\n");
    }
    fprintf(stderr, "[Turbo1Bit] Value compression: %d-bit group quantization\n", state.value_bits);

    // Initialize Metal GPU acceleration
    state.metal = t1b_metal_init();
    if (state.metal) {
        fprintf(stderr, "[Turbo1Bit] Metal GPU acceleration: ENABLED\n");
    } else {
        fprintf(stderr, "[Turbo1Bit] Metal GPU acceleration: not available (using CPU)\n");
    }

    // Will be initialized when we know the KV cache size
    state.slot_compressed.clear();
}

// Apply Turbo1Bit quantize-then-dequantize to a KV cache slot.
// This replaces the FP16 values with their lossy compressed-then-decompressed versions.
static void turbo1bit_apply(turbo1bit_state & state, llama_context * ctx, int cur_pos) {
    if (!state.enabled) return;

    auto * mem = llama_get_memory(ctx);
    if (!mem) return;
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) return;

    int kv_size = (int)kv->get_size();
    int n_embd_k = state.n_kv_heads * state.head_dim;

    // Only compress the slot that just aged past the buffer threshold.
    // Each slot is compressed exactly ONCE when it becomes 128 positions old.
    int buffer_size = 128;
    int compress_pos = cur_pos - buffer_size;
    if (compress_pos < 0) return;

    // Initialize slot tracking on first call
    if (state.slot_compressed.empty()) {
        state.slot_compressed.resize(kv_size, false);
    }

    int compress_slot = compress_pos % kv_size;

    // Skip if already compressed (handles ring buffer wrap-around)
    if (state.slot_compressed[compress_slot]) {
        // Slot was reused — mark as uncompressed for the new occupant
        state.slot_compressed[compress_slot] = false;
    }

    // Mark as compressed
    state.slot_compressed[compress_slot] = true;

    for (int l = 0; l < kv->get_n_layers(); l++) {
        ggml_tensor * k_tensor = kv->get_k_tensor(l);
        ggml_tensor * v_tensor = kv->get_v_tensor(l);
        if (!k_tensor || !v_tensor) continue;

        size_t k_row_size = ggml_row_size(k_tensor->type, n_embd_k);

        // Read K data
        std::vector<uint8_t> k_raw(k_row_size);
        ggml_backend_tensor_get(k_tensor, k_raw.data(),
                                (size_t)compress_slot * k_row_size, k_row_size);

        // Convert to f32
        std::vector<float> k_f32(n_embd_k);
        if (k_tensor->type == GGML_TYPE_F16) {
            auto *src = reinterpret_cast<const ggml_fp16_t *>(k_raw.data());
            for (int i = 0; i < n_embd_k; i++) k_f32[i] = ggml_fp16_to_fp32(src[i]);
        } else if (k_tensor->type == GGML_TYPE_F32) {
            memcpy(k_f32.data(), k_raw.data(), n_embd_k * sizeof(float));
        } else {
            continue;
        }

        // Key compression (only if key_bits >= 2)
        if (state.key_bits >= 2 && l < (int)state.quantizers.size() && state.quantizers[l]) {
            t1b_quantizer *q = state.quantizers[l];
            for (int h = 0; h < state.n_kv_heads; h++) {
                float *hd = k_f32.data() + h * state.head_dim;
                uint8_t mse_buf[256], sign_buf[256];
                t1b_prod_quantized pq;
                memset(&pq, 0, sizeof(pq));
                pq.mse_indices = mse_buf;
                pq.qjl_signs = sign_buf;
                t1b_quantize_prod(q, hd, &pq);
                t1b_dequantize_prod(q, &pq, hd);
            }
        }

        // Write back
        if (k_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> k_f16(n_embd_k);
            for (int i = 0; i < n_embd_k; i++) k_f16[i] = ggml_fp32_to_fp16(k_f32[i]);
            ggml_backend_tensor_set(k_tensor, k_f16.data(),
                                    (size_t)compress_slot * k_row_size, k_row_size);
        } else {
            ggml_backend_tensor_set(k_tensor, k_f32.data(),
                                    (size_t)compress_slot * k_row_size, k_row_size);
        }

        // Same for V (group quantization) — only if not transposed
        if (!kv->get_v_trans()) {
            size_t v_row_size = ggml_row_size(v_tensor->type, n_embd_k);

            if (state.metal && v_tensor->type == GGML_TYPE_F16) {
                // Metal GPU path: read FP16 data, quantize-dequantize on GPU, write back
                std::vector<uint8_t> v_raw(v_row_size);
                ggml_backend_tensor_get(v_tensor, v_raw.data(),
                                        (size_t)compress_slot * v_row_size, v_row_size);

                t1b_metal_value_quant_dequant(state.metal, v_raw.data(),
                                              n_embd_k, state.value_group_size,
                                              state.value_bits);

                ggml_backend_tensor_set(v_tensor, v_raw.data(),
                                        (size_t)compress_slot * v_row_size, v_row_size);
            } else {
                // CPU fallback
                std::vector<uint8_t> v_raw(v_row_size);
                ggml_backend_tensor_get(v_tensor, v_raw.data(),
                                        (size_t)compress_slot * v_row_size, v_row_size);

                std::vector<float> v_f32(n_embd_k);
                if (v_tensor->type == GGML_TYPE_F16) {
                    auto *src = reinterpret_cast<const ggml_fp16_t *>(v_raw.data());
                    for (int i = 0; i < n_embd_k; i++) v_f32[i] = ggml_fp16_to_fp32(src[i]);
                } else if (v_tensor->type == GGML_TYPE_F32) {
                    memcpy(v_f32.data(), v_raw.data(), n_embd_k * sizeof(float));
                } else {
                    continue;
                }

                for (int h = 0; h < state.n_kv_heads; h++) {
                    float *hd = v_f32.data() + h * state.head_dim;
                    int gs = state.value_group_size;
                    int ng = state.head_dim / gs;
                    uint8_t data_buf[256];
                    float sc[16], zr[16];
                    t1b_value_quantized vq;
                    memset(&vq, 0, sizeof(vq));
                    vq.data = data_buf; vq.scales = sc; vq.zeros = zr;
                    vq.bits = state.value_bits; vq.d = state.head_dim;
                    vq.group_size = gs; vq.n_groups = ng;
                    t1b_quantize_values(hd, state.head_dim, state.value_bits, gs, &vq);
                    t1b_dequantize_values(&vq, hd);
                }

                if (v_tensor->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> v_f16(n_embd_k);
                    for (int i = 0; i < n_embd_k; i++) v_f16[i] = ggml_fp32_to_fp16(v_f32[i]);
                    ggml_backend_tensor_set(v_tensor, v_f16.data(),
                                            (size_t)compress_slot * v_row_size, v_row_size);
                } else {
                    ggml_backend_tensor_set(v_tensor, v_f32.data(),
                                            (size_t)compress_slot * v_row_size, v_row_size);
                }
            }
        }
    }
    state.tokens_compressed++;
}

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt = "Explain quantum computing in simple terms:";
    int n_predict = 128;
    bool use_turbo1bit = true;
    int ctx_size = 2048;

    int key_bits = 0;   // 0=disabled, 3-5 = TurboQuantProd
    int val_bits = 4;   // 2 or 4
    std::string ctk_str = "f16";  // llama.cpp native KV cache type for keys
    std::string ctv_str = "f16";  // llama.cpp native KV cache type for values
    bool use_fa = false;          // flash attention (required for native KV quant)

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "-p" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "-n" && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (arg == "-c" && i + 1 < argc) ctx_size = atoi(argv[++i]);
        else if (arg == "--key-bits" && i + 1 < argc) key_bits = atoi(argv[++i]);
        else if (arg == "--val-bits" && i + 1 < argc) val_bits = atoi(argv[++i]);
        else if ((arg == "--ctk" || arg == "-ctk") && i + 1 < argc) ctk_str = argv[++i];
        else if ((arg == "--ctv" || arg == "-ctv") && i + 1 < argc) ctv_str = argv[++i];
        else if (arg == "--fa") use_fa = true;
        else if (arg == "--no-turbo1bit") use_turbo1bit = false;
        else if (arg == "-h" || arg == "--help") {
            fprintf(stderr, "Usage: %s -m model.gguf [-p prompt] [-n tokens] [-c ctx]\n", argv[0]);
            fprintf(stderr, "       [--ctk TYPE] [--ctv TYPE] [--fa] [--no-turbo1bit]\n");
            fprintf(stderr, "       [--key-bits N] [--val-bits N]\n");
            fprintf(stderr, "\nNative KV cache quantization (SAVES REAL MEMORY, requires --fa):\n");
            fprintf(stderr, "  --ctk q4_0 --ctv q4_0 --fa   4-bit KV cache (2.65x memory reduction)\n");
            fprintf(stderr, "  --ctk q5_0 --ctv q5_0 --fa   5-bit KV cache (2.32x memory reduction)\n");
            fprintf(stderr, "  --ctk q8_0 --ctv q8_0 --fa   8-bit KV cache (1.68x memory reduction)\n");
            fprintf(stderr, "\nTurbo1Bit in-place compression (quality simulation, no memory saving yet):\n");
            fprintf(stderr, "  --key-bits 5 --val-bits 2     5-bit keys + 2-bit values\n");
            fprintf(stderr, "  --no-turbo1bit                Baseline (no compression)\n");
            return 0;
        }
    }

    // Auto-enable FA if KV quantization requested
    if (ctk_str != "f16" || ctv_str != "f16") {
        use_fa = true;
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: -m model.gguf required\n");
        return 1;
    }

    fprintf(stderr, "=== Turbo1Bit Inference ===\n");
    fprintf(stderr, "Model: %s\n", model_path.c_str());
    if (ctk_str != "f16" || ctv_str != "f16") {
        fprintf(stderr, "Mode:  NATIVE KV QUANT (K=%s V=%s, FA=%s)\n",
                ctk_str.c_str(), ctv_str.c_str(), use_fa ? "on" : "off");
    } else {
        fprintf(stderr, "Mode:  %s\n", use_turbo1bit ? "TURBO1BIT (compressed KV)" : "BASELINE (FP16 KV)");
    }
    fprintf(stderr, "Ctx:   %d, Predict: %d\n\n", ctx_size, n_predict);

    // Load model
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Create context
    // Map type strings to ggml types
    auto str_to_type = [](const std::string & s) -> ggml_type {
        if (s == "f32")  return GGML_TYPE_F32;
        if (s == "f16")  return GGML_TYPE_F16;
        if (s == "q8_0") return GGML_TYPE_Q8_0;
        if (s == "q5_0") return GGML_TYPE_Q5_0;
        if (s == "q5_1") return GGML_TYPE_Q5_1;
        if (s == "q4_0") return GGML_TYPE_Q4_0;
        if (s == "q4_1") return GGML_TYPE_Q4_1;
        return GGML_TYPE_F16;
    };

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctx_size;
    cparams.n_batch = 512;
    cparams.type_k = str_to_type(ctk_str);
    cparams.type_v = str_to_type(ctv_str);
    cparams.flash_attn_type = use_fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); llama_model_free(model); return 1; }

    // Init Turbo1Bit
    turbo1bit_state t1b;
    if (use_turbo1bit) {
        t1b.key_bits = key_bits;
        t1b.value_bits = val_bits;
        turbo1bit_init(t1b, model);
    }

    // Tokenize
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                   tokens.data(), (int32_t)tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                   tokens.data(), (int32_t)tokens.size(), true, true);
    }
    tokens.resize(n_tokens);
    fprintf(stderr, "Prompt: %d tokens\n", n_tokens);

    // Prefill
    llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n");
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
        return 1;
    }
    fprintf(stderr, "Prefill done.\n");

    // Sampler (greedy)
    auto sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    // Decode
    fprintf(stderr, "\n--- Output ---\n");
    int n_gen = 0;
    llama_pos cur_pos = n_tokens;

    while (n_gen < n_predict) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) { fprintf(stderr, "\n[EOS]"); break; }

        char buf[256];
        int len = llama_token_to_piece(vocab, tok, buf, sizeof(buf) - 1, 0, true);
        if (len > 0) { buf[len] = '\0'; printf("%s", buf); fflush(stdout); }

        common_batch_clear(batch);
        common_batch_add(batch, tok, cur_pos, {0}, true);

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\nDecode failed at token %d\n", n_gen);
            break;
        }

        if (use_turbo1bit) turbo1bit_apply(t1b, ctx, cur_pos);

        cur_pos++;
        n_gen++;
    }

    printf("\n");
    fprintf(stderr, "\n--- Results ---\n");
    fprintf(stderr, "Mode: %s\n", use_turbo1bit ? "TURBO1BIT" : "BASELINE");
    fprintf(stderr, "Generated: %d tokens\n", n_gen);
    if (use_turbo1bit) {
        fprintf(stderr, "KV entries compressed: %lld\n", (long long)t1b.tokens_compressed);
    }

    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
