# Human vs. AI-Generated Text Detection
### NLP Class — ENSAE Paris, 2026
*Louis-Armand Aumont*

---

## Overview

Detecting LLM-generated texts has become increasingly challenging with the rapid 
improvement of large language models. This repository implements and evaluates 
**zero-shot detection methods** for identifying AI-generated text, without requiring 
any model training or access to the source model's architecture.

We first conduct a **corpus analysis** to extract stylometric characteristics of 
LLM-generated texts (Yule's K, MATTR, POS tagging), then implement and compare 
two state-of-the-art zero-shot detection algorithms: **DetectGPT** and **Binoculars**.

> **Best result:** Binoculars achieves an AUROC of **0.8745** on the test set.

## Methods

### Corpus Analysis
- **Lexical diversity:** MATTR (local) and Yule's K (global), both robust to text length
- **POS tagging:** Unigram and bigram analysis using spaCy `en_core_web_sm`
- **Statistical testing:** Mann-Whitney U test with Cohen's d effect sizes

### Zero-Shot Detection
| Method | Key idea | AUROC | Runtime (200 texts) |
|--------|----------|-------|---------------------|
| Log-probability | Single model likelihood | 0.748 | seconds |
| Perplexity | Length-normalized likelihood | 0.701 | seconds |
| DetectGPT | Log-prob curvature via perturbations | 0.752 | 25min 17s |
| **Binoculars** | **Cross-model likelihood disagreement** | **0.875** | **3min 21s** |


---

## Installation

```bash
git clone https://github.com/LouisArmand0/Detect_LLMs_generated_texts.git
cd Detect_LLMs_generated_texts
pip install -r requirements.txt
```

---
## References

- Mitchell et al. (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature.* ICML 2023.
- Hans et al. (2024). *Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated Text.* ICML 2024.