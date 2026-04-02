# LinkedIn Post

---

**We just ran an 8 billion parameter LLM at full 65K context on 3.9 GB of RAM.**

I open-sourced Turbo1Bit — a project that enables KV cache compression for PrismML's Bonsai 1-bit LLMs. The result: an 8B model that normally needs 10.4 GB now fits on an 8GB MacBook Air.

Measured results (Apple Silicon, RSS via /usr/bin/time):

Bonsai-8B at 65K context:
  Before: 10,618 MiB (10.4 GB) — doesn't fit on 8GB
  After:   4,000 MiB (3.9 GB)  — fits comfortably

Bonsai-1.7B at 65K context:
  Before:  7,846 MiB
  After:   2,694 MiB — saves 5 GB

How we got here:

Bonsai's 1-bit models are already incredibly small — the 8B model is just 1.1 GB on disk. But at long contexts, the KV cache dominates memory. At 65K tokens, the KV cache alone is ~9 GB in FP16.

We set out to port TurboQuant's ICLR 2026 compression algorithms (Lloyd-Max quantization, QJL sign projection) from Python to C. Along the way, we found that llama.cpp already has KV cache quantization (--ctk, --ctv flags) — but Bonsai's docs and scripts don't use it. Trying it without Flash Attention gives a cryptic error, so most users would assume it's unsupported.

The technical reason: without Flash Attention, llama.cpp stores the V cache transposed (one element per row), which breaks block quantization formats like Q4_0 that need groups of 32 contiguous values. Flash Attention stores V non-transposed, making quantization work.

Once we added --fa on, Q4_0 KV cache just worked. We then validated quality specifically for 1-bit models — something nobody had tested.

Flash Attention also turned out to be a 2.4x prefill speedup: 1,425 tok/s without FA vs 3,452 tok/s with FA. So the compressed version is actually faster than the original.

The quality sweep was revealing. We tested KV compression at multiple levels with real model output:

Q8_0 KV: identical to baseline
Q5_0 KV: minor rephrasings, fully coherent
Q4_0 KV: slight variation, fully coherent

We also tested our custom TurboQuant compression and found that 1-bit models are more sensitive to key quantization than FP16 models. 3-bit keys (which work great on FP16 models per the TurboQuant paper) cause output to degrade with 1-bit weights. You need at least 4-bit keys for coherent output. 2-bit values, however, are completely lossless.

What this means practically:

An 8B model with 65K token context window running on a $999 MacBook Air. The model weights are 1.1 GB (1-bit), the KV cache is 2.9 GB (Q4_0), leaving headroom for the OS and apps.

The tool is simple to use:
  ./turbo1bit run Bonsai-8B.gguf "Your prompt" -c 65536

It auto-detects your RAM and picks the best compression level.

Code + full benchmarks: https://github.com/jhammant/Turbo1bit

Built on:
- Bonsai by PrismML: https://github.com/PrismML-Eng/Bonsai-demo
- TurboQuant by 0xSero (ICLR 2026): https://github.com/0xSero/turboquant
- llama.cpp by Georgi Gerganov

#LLM #AI #MachineLearning #EdgeAI #Quantization #OpenSource #AppleSilicon #OnDeviceAI #Inference
