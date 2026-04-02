# Turbo1Bit

**Combining 1-bit LLM weights with compressed KV caches for maximum inference efficiency.**

Turbo1Bit merges two cutting-edge approaches to LLM efficiency:

- **[Bonsai](https://github.com/PrismML-Eng/Bonsai-demo)** (PrismML): 1-bit model weight quantization — reduces model memory by ~16x
- **[TurboQuant](https://github.com/0xSero/turboquant)** (ICLR 2026): KV cache compression with 3-bit keys + 2-bit values — reduces runtime memory by ~4.2x

Together, they achieve **~10x total memory reduction** for LLM inference on consumer hardware.

## How It Works

```text
Standard LLM Inference          Turbo1Bit
┌──────────────────┐            ┌──────────────────┐
│ Model Weights    │            │ Model Weights    │
│ FP16: ~3.4 GB    │ ──────▶   │ 1-bit: ~0.2 GB   │  (Bonsai)
│ (1.7B params)    │            │ (1.7B params)    │
├──────────────────┤            ├──────────────────┤
│ KV Cache @ 32K   │            │ KV Cache @ 32K   │
│ FP16: 3.0 GB     │ ──────▶   │ Compressed: 0.7 GB│  (TurboQuant)
│ (24 layers)      │            │ (24 layers)      │
├──────────────────┤            ├──────────────────┤
│ Total: ~6.4 GB   │            │ Total: ~0.9 GB   │  7x smaller
└──────────────────┘            └──────────────────┘
```

## Benchmark Results

### KV Cache Memory Compression (Bonsai-1.7B, 24 layers)

| Context Length | FP16 KV Cache | Turbo1Bit KV | Compression Ratio |
|---------------|--------------|-------------|-------------------|
| 1K | 96 MB | 68 MB | 2.2x |
| 8K | 768 MB | 225 MB | 3.8x |
| 32K | 3,072 MB | 765 MB | 4.1x |
| 65K | 6,144 MB | 1,485 MB | 4.2x |
| **131K** | **12,288 MB** | **2,925 MB** | **4.24x** |

### Maximum Context Lengths (model weights + KV cache)

| Model | Hardware | FP16 Max Context | Turbo1Bit Max | Improvement |
|-------|----------|-----------------|---------------|-------------|
| Bonsai-1.7B | 8 GB M1 | 72K tokens | **290K tokens** | 4x |
| Bonsai-1.7B | 16 GB M1 Pro | 157K tokens | **631K tokens** | 4x |
| Bonsai-8B | 16 GB M1 Pro | 112K tokens | **448K tokens** | 4x |
| Bonsai-8B | 32 GB M1 Max | 240K tokens | **960K tokens** | 4x |

### Quantization Quality

| Component | Cosine Similarity | Notes |
|-----------|------------------|-------|
| 3-bit key compression | 0.920 | Per-vector fidelity |
| 2-bit value quantization | 0.961 | Group quantization |
| 4-bit value quantization | 0.998 | High-quality option |
| Rotation orthogonality | 1.000 | Mathematically exact |

### Per-Token Storage Breakdown

```text
                    FP16        Turbo1Bit     Savings
Key (per head):     256 bytes   56 bytes      4.6x
Value (per head):   256 bytes   64 bytes      4.0x
Combined:           512 bytes   120 bytes     4.27x
```

## Architecture

Turbo1Bit implements TurboQuant's compression pipeline in C for integration with llama.cpp:

1. **Lloyd-Max Codebook Quantization**: Optimal scalar quantizer for rotated coordinates
2. **Random Orthogonal Rotation**: Decorrelates dimensions before quantization (via QR decomposition)
3. **QJL Sign Projection**: Captures residual information in 1 bit per dimension
4. **Group Quantization**: Per-group min-max quantization for value vectors
5. **Bit Packing**: Dense storage of compressed representations

```text
Key Compression (TurboQuantProd, 3 bits total):
  x → normalize → rotate(Pi) → Lloyd-Max(2-bit) → bit-pack   [MSE stage]
  x → reconstruct → residual → project(S) → sign() → pack    [QJL stage]

Value Compression (Group Quantization, 2 bits):
  v → group(32) → per-group min-max → quantize(2-bit) → pack
```

## Building

```bash
# Clone with submodules
git clone https://github.com/jhammant/Turbo1bit.git
cd Turbo1bit

# Clone dependencies
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp
git clone --depth 1 https://github.com/0xSero/turboquant.git

# Build (macOS with Metal)
cd bonsai-llama.cpp
mkdir build && cd build
cmake .. -G Ninja -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
ninja turbo1bit-bench turbo1bit-stress

# Run benchmarks
./bin/turbo1bit-bench      # Core tests
./bin/turbo1bit-stress     # Stress test with extreme context lengths
```

## Project Structure

```text
Turbo1bit/
├── bonsai-llama.cpp/src/
│   ├── turbo1bit_codebook.h/c      # Pre-computed Lloyd-Max codebooks
│   ├── turbo1bit_rotation.h/c      # Orthogonal rotation + QJL matrices
│   ├── turbo1bit_quantizer.h/c     # MSE + Prod quantizers + value quant
│   ├── turbo1bit_kv_cache.h/c      # Compressed KV cache manager
│   ├── turbo1bit_metal.h/m         # Metal GPU acceleration (Apple Silicon)
│   └── turbo1bit_metal.metal       # Metal compute shaders
├── bonsai-llama.cpp/tools/turbo1bit/
│   ├── turbo1bit_bench.c           # Core benchmark suite
│   ├── turbo1bit_stress.c          # Extreme context stress test
│   └── turbo1bit_debug.c           # Quality debugging
├── benchmark.sh                     # End-to-end comparison script
└── README.md
```

## How the Compression Works

### Key Compression: TurboQuantProd (3 bits)

The key insight from the TurboQuant paper (ICLR 2026) is that after random orthogonal rotation, each coordinate of a unit-norm vector follows a known distribution. This allows optimal Lloyd-Max quantization with pre-computed codebooks.

**Stage 1 — MSE Quantization (2 bits):**
- Normalize key vector to unit sphere
- Apply random orthogonal rotation (decorrelates coordinates)
- Quantize each coordinate using 4-centroid Lloyd-Max codebook
- Bit-pack: 4 values per byte

**Stage 2 — QJL Residual (1 bit):**
- Compute residual: r = key - MSE_reconstruction
- Project through random Gaussian matrix S
- Store sign of each projection: 8 signs per byte
- Store residual norm for rescaling

**Attention scoring** uses the asymmetric estimator:
```text
<query, key> ≈ <query, key_mse> + (√(π/2)/d) · ||residual|| · <query·S^T, signs>
```

### Value Compression: Group Quantization (2 bits)

- Divide each value vector into groups of 32 elements
- Per-group asymmetric quantization: scale = (max - min) / 3
- Store as 2-bit indices (4 per byte) + per-group scale and zero point

## Credits

- **[TurboQuant](https://github.com/0xSero/turboquant)** by 0xSero — KV cache compression algorithms (ICLR 2026)
- **[Bonsai / PrismML](https://github.com/PrismML-Eng/Bonsai-demo)** — 1-bit LLM models and inference
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — C/C++ LLM inference engine

## License

GPL-3.0 (following TurboQuant's license)
