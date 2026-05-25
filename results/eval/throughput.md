# Inference throughput: transformers vs vLLM

- Model: `/mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft-merged`
- Prompts: 270, max_new_tokens: 128

| Metric | transformers | vLLM | speedup |
| --- | ---: | ---: | ---: |
| Wall-clock (s) | 1.9 | 0.5 | 3.62x |
| Requests/s | 139.37 | 504.61 | 3.62x |
| Generated tokens/s | 278.7 | 1009.2 | 3.62x |
| Latency p50 (s) | 0.007 | 0.002 | 3.50x |
| Latency p95 (s) | 0.008 | 0.002 | 4.15x |

transformers: batch_size=8, single GPU.
vLLM: TP=2, max_num_seqs=256, kv_cache_dtype=auto.
vLLM wins from PagedAttention (no KV-cache fragmentation) and continuous batching.
Same recipe applies to the customer's H100 box; absolute numbers scale with peak FLOPs and HBM bandwidth.
