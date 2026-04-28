# Effect of Input Modality on Mathematical Reasoning in Vision-Language Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOURUSERNAME/vlm-gsm8k-benchmark/blob/main/notebook/VLM_GSM8K_Benchmarking.ipynb)

Presented at the **USF UR2PhD Symposium 2025** · [View Poster](poster/VLM_GSM8K_Poster.pdf)

---

## Overview

This project investigates how input modality affects mathematical reasoning in vision-language models (VLMs). Using 100 problems from the GSM8K benchmark, we evaluate two models across three controlled conditions — text-only, rendered image, and a novel modality mismatch condition — to isolate the effect of input format on reasoning performance.

**Key findings:**
- Visual input causes substantial accuracy drops in both models (−25pp for Qwen2, −19pp for LLaVA)
- Arithmetic errors increase significantly under image input; vision/OCR errors remain negligible
- In the mismatch condition, both models follow the text modality in over 75% of cases, revealing strong implicit text dominance

---

## Models

| Model | Size | Type |
|-------|------|------|
| Qwen2-VL-2B-Instruct | 2B | Open-source, local |
| LLaVA-v1.6-Mistral-7B | 7B | Open-source, local |

---

## Experimental Conditions

| Condition | Input | Vision Encoder |
|-----------|-------|----------------|
| 1 — Text-Only | Raw problem text | Disabled |
| 2 — Rendered Image | Clean PIL-rendered PNG | Enabled |
| 3 — Modality Mismatch | Image of problem i + text of problem i+1 | Enabled |

---

## Results

**Table 1: Zero-Shot Accuracy**

| Model | Text-Only | Rendered Image | Drop |
|-------|-----------|----------------|------|
| Qwen2-VL-2B | 55% | 30% | −25pp |
| LLaVA-1.6-7B | 40% | 21% | −19pp |

**Table 2: Modality Preference Under Conflict (n=100)**

| Model | Follows Image | Follows Text | Equal |
|-------|--------------|--------------|-------|
| Qwen2-VL-2B | 20 | 79 | 1 |
| LLaVA-1.6-7B | 23 | 76 | 0 |

---

## Repository Structure

```
vlm-gsm8k-benchmark/
├── README.md
├── notebooks/
│   ├── VLM_GSM8K_Benchmarking.ipynb   
├── rendered_images   
├── results/
│   ├── Qwen2-VL-2B-Instruct/
│   │   ├── gsm8k_vlm_results.csv
│   │   ├── error_summary.csv
│   │   └── mismatch_results.csv
│   └── LLaVa_1.6/
│       ├── gsm8k_vlm_results.csv
│       ├── error_summary.csv
│       └── mismatch_results.csv
└── poster/
    └── VLM_GSM8K_Poster.pdf
```

---

## How to Run

1. Open the relevant notebook in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Upload your pre-rendered images to `/content/images/` in the Colab file browser
4. Set `IMAGE_DIR = "/content/images"` in the config cell
5. Run all cells sequentially

**Dependencies are installed automatically by the first cell.**

---

## Evaluation

- Numeric answer extracted using GSM8K canonical format (`#### <answer>`) with fallback to last number in output
- Answers compared by rounding to nearest integer
- Error classification: `correct`, `arithmetic_error`, `reasoning_error`, `no_number`, `vision_error`
- Mismatch condition scored by absolute numeric distance to each modality's ground truth

---

## Limitations

- 100 problems is a small sample — findings are indicative rather than statistically conclusive
- Both models are relatively small (2B and 7B); behaviour may differ at larger scales
- Rendered images use clean digital text — real-world visual noise not tested
- Evaluation on GSM8K only; generalisability to other math benchmarks not confirmed

---

## Future Work

- Scale to the full GSM8K test set for statistical reliability
- Add significance testing (McNemar's test, chi-squared)
- Evaluate frontier models (GPT-4o, Gemini 2.0) to assess whether text dominance persists at scale
- Implement screenshot condition with controlled noise levels
- Test on additional benchmarks (SVAMP, AQuA-RAT)

---

## References

[1] Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.  
[2] Wang et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191.  
[3] Liu et al. (2024). LLaVA-NeXT: Improved Baselines with Visual Instruction Tuning. arXiv:2310.03744.

---

## Authors

**Rodela Ghosh** · University of South Florida  
**Aviral Gupta** · University of South Florida

---

## Acknowledgements

This project was conducted through the PALM Lab at the University of South Florida under the UR2PhD program. The authors thank graduate mentors Ocean Monjur and Shrestha Datta for their guidance and mentorship. We also thank PI Anshuman Chhabra and the UR2PhD team for providing research training, resources, and the opportunity to present this work.
