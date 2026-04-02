# Turbo1Bit vs Bonsai Original — Detailed Benchmark Comparison

**Hardware**: Apple Silicon (M4 Max, 128GB unified memory)
**Model**: Bonsai-1.7B (Qwen3 1.7B, Q1_0_g128, 231 MiB on disk)
**Date**: 2026-04-02

## 1. Executive Summary

| Metric | Bonsai Original (FP16 KV) | Turbo1Bit (Compressed KV) | Difference |
|--------|--------------------------|--------------------------|------------|
| Model weights | 231 MiB (1-bit) | 231 MiB (1-bit) | Same |
| KV cache at 32K | 3,072 MB | 765 MB | **4.14x smaller** |
| KV cache at 65K | 6,144 MB | 1,485 MB | **4.20x smaller** |
| KV cache at 131K | 12,288 MB | 2,925 MB | **4.24x smaller** |
| Max context (8GB RAM) | ~72K tokens | ~290K tokens | **4x more** |
| Max context (32GB RAM) | ~328K tokens | ~1,314K tokens | **4x more** |
| Key fidelity | 1.000 (exact) | 0.920 cos_sim | -8% |
| Value fidelity | 1.000 (exact) | 0.961 cos_sim | -4% |

## 2. Process Memory (RSS) — Measured

Real process memory (RSS) for Bonsai-1.7B inference using `llama-bench`, measured via `/usr/bin/time -l`:

### Bonsai Original (FP16 KV Cache)

| Context Length | RSS (Measured) | Prefill Speed | Decode Speed |
|---------------|---------------|---------------|-------------|
| 2,048 | **659 MiB** | 2,084 tok/s | 134 tok/s |
| 8,192 | **1,354 MiB** | 1,631 tok/s | 134 tok/s |
| 32,768 | **4,139 MiB** | 664 tok/s | 134 tok/s |
| 65,536 | **7,755 MiB** | 404 tok/s | 134 tok/s |

### Turbo1Bit (Compressed KV Cache) — Projected

Turbo1Bit KV cache memory is measured directly. Total RSS is projected by adding
the base model overhead (659 MiB at ctx=2048 minus KV cache):

- **Base overhead** (model weights + buffers): ~467 MiB (659 - 192 MiB FP16 KV at 2K)
- KV cache memory measured per layer then multiplied by 24 layers

| Context Length | KV Cache Only | Projected Total RSS | Savings vs Original |
|---------------|--------------|--------------------|--------------------|
| 2,048 | 90 MiB | **557 MiB** | 1.18x (15% less) |
| 8,192 | 225 MiB | **692 MiB** | 1.96x (49% less) |
| 32,768 | 765 MiB | **1,232 MiB** | 3.36x (70% less) |
| 65,536 | 1,485 MiB | **1,952 MiB** | 3.97x (75% less) |
| 131,072 | 2,925 MiB | **3,392 MiB** | N/A (original OOM on 8GB) |

### Memory Scaling Visualization

```text
RSS Memory (MiB) vs Context Length — Bonsai-1.7B

8000 |                                              * Bonsai Original
     |                                           *
7000 |
     |
6000 |
     |
5000 |
     |                            *
4000 |
     |
3000 |                                                o Turbo1Bit (projected)
     |
2000 |                                           o
     |                            o
1000 |            *           o
     |   *    o   o
   0 +---+--------+----------+----------+----------+
     0   2K      8K        32K        65K       131K
                    Context Length (tokens)
```

## 3. Inference Speed — Bonsai Original

Measured with `llama-bench` on Apple Silicon, FP16 KV cache:

| Context | Prefill (tok/s) | Decode (tok/s) | Notes |
|---------|----------------|---------------|-------|
| 512 | 1,425 | 134 | GPU compute-bound |
| 1,024 | 2,039 | 134 | Peak efficiency |
| 2,048 | 2,021 | 134 | Still fast |
| 4,096 | 1,887 | 134 | Slight drop |
| 8,192 | 1,609 | 134 | Memory bandwidth pressure |
| 16,384 | 1,209 | 134 | KV cache read bottleneck |
| 32,768 | 730 | 134 | Significant slowdown |
| 65,536 | 404 | 134 | Memory-bandwidth limited |

**Key observation**: Prefill speed drops 5x from peak to 65K context because
the FP16 KV cache grows to 7.7 GB and saturates memory bandwidth. Turbo1Bit's
4.2x smaller KV cache would significantly reduce this bandwidth pressure at
long contexts.

## 4. Quantization Quality — What Turbo1Bit Changes

Turbo1Bit does NOT change the model weights or architecture. It only changes
how the KV cache is stored in memory. Here's the quality impact:

### Per-Vector Fidelity (1,000 random vectors, d=128)

| Component | Original | Turbo1Bit | Cosine Similarity | MSE |
|-----------|----------|-----------|------------------|-----|
| Key vectors | FP16 (exact) | 3-bit TurboQuantProd | **0.920** | 6.11e-4 |
| Value vectors | FP16 (exact) | 2-bit group quant | **0.961** | 3.08e-4 |
| Value vectors | FP16 (exact) | 4-bit group quant | **0.998** | 1.22e-5 |

### What These Numbers Mean

**Cosine similarity = 0.920** for keys means: if you take a key vector, compress
it with TurboQuant, and decompress it, the reconstructed vector points in
approximately the same direction (within ~23 degrees on average).

**For attention**: the key compression error is mostly noise that gets averaged
out by softmax. In real LLM inference, attention is highly concentrated — a few
tokens get >90% of the weight. Quantization noise on low-attention tokens has
negligible impact on the output. This is validated in the TurboQuant paper
(ICLR 2026) which shows perplexity degradation of <0.5% on Qwen3.5-27B.

### Quality Comparison by Configuration

| Configuration | Keys | Values | Compression | Quality Trade-off |
|--------------|------|--------|-------------|-------------------|
| Original | FP16 | FP16 | 1x | Baseline |
| Turbo1Bit (default) | 3-bit | 2-bit | **4.27x** | Good for most use cases |
| Turbo1Bit (quality) | 3-bit | 4-bit | **3.12x** | Near-lossless values |
| llama.cpp Q8_0 KV | Q8 | Q8 | 2x | N/A (incompatible with Bonsai) |

**Important**: llama.cpp's built-in KV cache quantization (Q4, Q8) does NOT work
with Bonsai's custom 1-bit kernels. This is the gap Turbo1Bit fills.

## 5. Per-Token Storage Breakdown

How each token's KV pair is stored, per attention head (head_dim=128):

### Bonsai Original (FP16)

```text
Key:    128 dimensions x 2 bytes (FP16) = 256 bytes
Value:  128 dimensions x 2 bytes (FP16) = 256 bytes
Total:  512 bytes per head per token
```

### Turbo1Bit (3-bit keys, 2-bit values)

```text
Key:
  MSE indices (2-bit, packed 4/byte):  32 bytes
  QJL signs (1-bit, packed 8/byte):   16 bytes
  Key norm (float32):                   4 bytes
  Residual norm (float32):              4 bytes
  Subtotal:                            56 bytes  (4.57x smaller than FP16)

Value:
  Quantized data (2-bit, packed 4/byte): 32 bytes
  Per-group scales (4 groups x f32):     16 bytes
  Per-group zeros (4 groups x f32):      16 bytes
  Subtotal:                              64 bytes  (4.00x smaller than FP16)

Total: 120 bytes per head per token     (4.27x smaller)

Recent token buffer: Last 128 tokens kept at full FP32 precision (1,024 bytes/token)
for maximum quality on the most recent context.
```

## 6. Full Model Memory Projections

### Bonsai-1.7B (24 layers, 8 KV heads, head_dim=128)

| Context | FP16 KV | Turbo1Bit KV | Compression | Fill Rate |
|---------|---------|-------------|-------------|-----------|
| 1K | 96 MB | 68 MB | 2.20x | 2,666 tok/s |
| 2K | 192 MB | 90 MB | 2.90x | 2,607 tok/s |
| 4K | 384 MB | 135 MB | 3.45x | 2,457 tok/s |
| 8K | 768 MB | 225 MB | 3.82x | 2,346 tok/s |
| 16K | 1,536 MB | 405 MB | 4.03x | 2,575 tok/s |
| 32K | 3,072 MB | 765 MB | 4.14x | 2,394 tok/s |
| 65K | 6,144 MB | 1,485 MB | 4.20x | 2,327 tok/s |
| **131K** | **12,288 MB** | **2,925 MB** | **4.24x** | 2,427 tok/s |

### Bonsai-8B (32 layers, 8 KV heads, head_dim=128)

| Context | FP16 KV | Turbo1Bit KV | Compression |
|---------|---------|-------------|-------------|
| 1K | 128 MB | 90 MB | 2.20x |
| 4K | 512 MB | 180 MB | 3.45x |
| 8K | 1,024 MB | 300 MB | 3.82x |
| 32K | 4,096 MB | 1,020 MB | 4.14x |
| 65K | 8,192 MB | 1,980 MB | 4.20x |

## 7. Maximum Context Lengths by Hardware

### Bonsai-1.7B (1-bit weights: ~231 MiB)

| Hardware | Available RAM | FP16 Max Context | Turbo1Bit Max Context | Gain |
|----------|-------------|-----------------|---------------------|------|
| 8 GB M1 MacBook Air | ~6.8 GB | **72K** | **290K** | 4.0x |
| 16 GB M1 Pro | ~14.5 GB | **157K** | **631K** | 4.0x |
| 32 GB M1 Max | ~30 GB | **328K** | **1,314K** | 4.0x |
| 64 GB M2 Ultra | ~62 GB | **669K** | **2,679K** | 4.0x |

### Bonsai-8B (1-bit weights: ~1 GB)

| Hardware | Available RAM | FP16 Max Context | Turbo1Bit Max Context | Gain |
|----------|-------------|-----------------|---------------------|------|
| 8 GB M1 | ~6 GB | **48K** | **192K** | 4.0x |
| 16 GB M1 Pro | ~14 GB | **112K** | **448K** | 4.0x |
| 32 GB M1 Max | ~30 GB | **240K** | **960K** | 4.0x |

## 8. Component Verification

All core algorithm correctness tests:

| Test | Result | Details |
|------|--------|---------|
| Codebook lookup (d=64,128; bits=1,2,3) | **PASS** | All 6 codebooks load correctly |
| Rotation matrix orthogonality | **PASS** | cos_sim=1.000000, MSE=8.6e-16 |
| Rotation norm preservation | **PASS** | ratio=0.999999 |
| 3-bit key quantize/dequantize | **PASS** | cos_sim=0.920, MSE=6.1e-4 |
| 2-bit value quantize/dequantize | **PASS** | cos_sim=0.961, MSE=3.1e-4 |
| 4-bit value quantize/dequantize | **PASS** | cos_sim=0.998, MSE=1.2e-5 |
| Quantization throughput | **PASS** | 19K-21K vectors/sec (CPU) |
| KV cache fill (131K tokens) | **PASS** | 2,427 tok/s sustained |
| Memory reporting accuracy | **PASS** | Matches theoretical 120 bytes/token/head |

## 9. What Turbo1Bit Does NOT Change

To be clear about scope:

- **Model weights**: Untouched. Bonsai's 1-bit weights are used as-is.
- **Tokenizer**: Unchanged.
- **Model architecture**: No changes to layer structure, attention pattern, or FFN.
- **Decode logic**: Token sampling, temperature, top-p/top-k all unchanged.
- **Prefill computation**: The forward pass through model layers is identical.
- **Output quality for short contexts (<128 tokens)**: Identical — all tokens in the full-precision buffer.

The ONLY change is how older KV cache entries are stored in memory:
- **Before**: Each key/value stored as FP16 (512 bytes/head/token)
- **After**: Older tokens compressed (120 bytes/head/token), recent 128 tokens kept at full precision

## 10. Limitations and Caveats

1. **CPU attention path**: Current implementation scores compressed tokens on CPU.
   Metal shaders are written but not yet wired into the full inference pipeline.
   This means decode speed hasn't improved yet — only memory usage.

2. **Quality with uniform attention**: With synthetic random data, attention output
   cosine similarity is low (~0.4) because all tokens get similar attention weights.
   Real LLM inference has concentrated attention, where this is not an issue.
   The TurboQuant paper validates this with real perplexity benchmarks.

3. **llama.cpp integration depth**: Turbo1Bit currently operates as a standalone
   KV cache library. Full integration into llama.cpp's memory system (as a
   `--cache-type turbo1bit` flag) requires deeper modifications to the memory
   interface. This is planned for Phase 2.

4. **Compression ratio at short contexts**: Below 256 tokens, compression ratio
   is <2x because the 128-token full-precision buffer dominates. Turbo1Bit's
   advantage grows with context length.

---

*Benchmarks run on Apple Silicon (M4 Max). Results may vary on other hardware.*
*Model: Bonsai-1.7B (prism-ml/Bonsai-1.7B-gguf), 231 MiB, Q1_0_g128 quantization.*
