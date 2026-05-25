# Accuracy: base vs SFT

- Base: `Qwen/Qwen2.5-7B-Instruct`
- SFT:  `/mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft-merged`

| Task | Base acc | SFT acc | Delta (pp) |
| --- | ---: | ---: | ---: |
| mmlu_college_biology | 0.8403 | 0.8611 | +2.08 |
| **mmlu_high_school_mathematics** | 0.4296 | 0.5630 | +13.33 |
| mmlu_high_school_us_history | 0.8824 | 0.8725 | -0.98 |
| mmlu_world_religions | 0.8713 | 0.8538 | -1.75 |
