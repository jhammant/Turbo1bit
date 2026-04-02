# Turbo1Bit

**Run Bonsai-8B at full 65K context on an 8GB MacBook Air.**

Turbo1Bit enables KV cache compression for [PrismML's Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) 1-bit LLMs. By combining Flash Attention with quantized KV storage, it reduces inference memory by up to **2.65x** — making large models fit on small hardware.

## Headline Result (Measured)

**Bonsai-8B (8.2B parameters, 1-bit weights, 1.1 GB on disk)**

| Context | Without Turbo1Bit | With Turbo1Bit | Saved |
|---------|------------------|---------------|-------|
| 8K | 2,379 MiB | 1,557 MiB | 822 MiB |
| 32K | 5,891 MiB | 2,626 MiB | **3.3 GB** |
| **65K** | **10,618 MiB** | **4,000 MiB** | **6.5 GB** |

At 65K context, Bonsai-8B needs 10.4 GB — too large for 8GB hardware. With Turbo1Bit, it fits in **3.9 GB**.

## Quick Start

```bash
# Clone and build
git clone https://github.com/jhammant/Turbo1bit.git
cd Turbo1bit
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git bonsai-llama.cpp
cd bonsai-llama.cpp && mkdir build && cd build
cmake .. -G Ninja -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
ninja turbo1bit-infer llama-bench
cd ../..

# Download a model
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('prism-ml/Bonsai-8B-gguf', local_dir='models/Bonsai-8B-gguf', allow_patterns='*.gguf')"

# Run with auto-optimized settings
./turbo1bit run models/Bonsai-8B-gguf/Bonsai-8B.gguf "Explain quantum computing:" -n 200 -c 8192

# Or use turbo1bit-infer directly
./bonsai-llama.cpp/build/bin/turbo1bit-infer \
    -m models/Bonsai-8B-gguf/Bonsai-8B.gguf \
    -p "Explain quantum computing:" \
    -n 200 -c 65536 \
    --ctk q4_0 --ctv q4_0
```

## How It Works

Turbo1Bit discovered that Bonsai's custom 1-bit inference kernels support quantized KV cache storage — but **only when Flash Attention is enabled**. Without FA, any KV quantization attempt fails with:

```text
llama_init_from_model: quantized V cache was requested, but this requires Flash Attention
```

With `--fa on`, llama.cpp's Q4_0/Q5_0/Q8_0 KV cache types work perfectly. This was not documented anywhere in the Bonsai project.

Flash Attention also provides a **2.4x prefill speedup** as a bonus:

| Mode | Prefill (tok/s) | Decode (tok/s) |
|------|----------------|---------------|
| No FA (original) | 1,425 | 134 |
| FA + FP16 KV | **3,452** | **151** |
| FA + Q4_0 KV | **3,435** | **131** |

## Full Benchmark Results

### Bonsai-1.7B (Measured RSS via `/usr/bin/time -l`)

| Context | FP16 | Q8_0 | Q5_0 | Q4_0 |
|---------|------|------|------|------|
| 2K | 648 MiB | 543 MiB | 501 MiB | 487 MiB |
| 8K | 1,344 MiB | 924 MiB | 756 MiB | 700 MiB |
| 32K | 4,131 MiB | 2,454 MiB | 1,780 MiB | 1,555 MiB |
| 65K | 7,846 MiB | 4,489 MiB | 3,142 MiB | 2,694 MiB |

### Bonsai-8B (Measured RSS)

| Context | FP16 | Q4_0 | Saved |
|---------|------|------|-------|
| 2K | 1,592 MiB | 1,293 MiB | 299 MiB |
| 8K | 2,379 MiB | 1,557 MiB | 822 MiB |
| 32K | 5,891 MiB | 2,626 MiB | 3,265 MiB |
| 65K | 10,618 MiB | 4,000 MiB | 6,618 MiB |

### Output Quality

All quantization levels produce coherent text. Tested with greedy sampling (temp=0) across multiple prompts:

| Config | Quality |
|--------|---------|
| FP16 (baseline) | Reference |
| Q8_0 + FA | Identical to baseline |
| Q5_0 + FA | Minor rephrasings, fully coherent |
| Q4_0 + FA | Slight variation, fully coherent |

## What Turbo1Bit Includes

| Component | Description |
|-----------|-------------|
| `turbo1bit` | Simple wrapper script with auto RAM detection |
| `turbo1bit-infer` | Non-interactive inference tool with KV compression flags |
| TurboQuant C port | Lloyd-Max codebooks, orthogonal rotation, QJL projection, group quantization |
| Metal shaders | 5 GPU kernels for compressed KV attention (Apple Silicon) |
| Quality sweep | Tested key compression at 3/4/5 bits, value compression at 2/4 bits |
| Benchmark suite | `turbo1bit-bench`, `turbo1bit-stress` for standalone KV cache testing |

### TurboQuant Compression Research

Beyond the native KV quantization, Turbo1Bit includes a C port of the [TurboQuant](https://github.com/0xSero/turboquant) (ICLR 2026) compression algorithms. Key findings for 1-bit models:

- **2-bit value compression**: Lossless — output identical to baseline
- **4-bit key compression**: Good quality — coherent text with minor rephrasings
- **3-bit key compression**: Too aggressive — output degrades to gibberish
- **Threshold**: 1-bit models need >= 4-bit keys (FP16 models can use 3-bit)

## Project Structure

```text
Turbo1bit/
├── turbo1bit                    # Simple wrapper script
├── src/                         # TurboQuant C port
│   ├── turbo1bit_codebook.h/c   # Lloyd-Max optimal codebooks
│   ├── turbo1bit_rotation.h/c   # QR rotation + QJL projection
│   ├── turbo1bit_quantizer.h/c  # MSE + Prod quantizers
│   ├── turbo1bit_kv_cache.h/c   # Compressed KV cache manager
│   ├── turbo1bit_metal.h/m      # Metal GPU host code
│   └── turbo1bit_metal.metal    # Metal compute shaders
├── tools/turbo1bit/
│   ├── turbo1bit_infer.cpp      # End-to-end inference tool
│   ├── turbo1bit_bench.c        # Core algorithm benchmarks
│   └── turbo1bit_stress.c       # Extreme context stress test
├── BENCHMARKS.md                # Detailed benchmark data
└── CMakeLists.txt               # Standalone build
```

## Credits

- **[Bonsai / PrismML](https://github.com/PrismML-Eng/Bonsai-demo)** — 1-bit LLM models and inference
- **[TurboQuant](https://github.com/0xSero/turboquant)** by 0xSero — KV cache compression algorithms (ICLR 2026)
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — C/C++ LLM inference engine

## License

GPL-3.0
