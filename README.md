# Toxic code review comment detection (tuned TF-IDF)

Compares a CV-tuned, class-weighted word+char TF-IDF logistic regression (optionally blended with LightGBM, with a CV-tuned decision threshold) against a plain word TF-IDF logistic regression on the ToxiCR code-review toxicity dataset, subsampled to ~10% toxic prevalence.

- Single stratified 80/20 split, seed 42; all tuning uses 5-fold CV on the training set only.
- Metric: toxic-class F1 on the held-out test set (also PR-AUC and toxic recall).
- Run `python analysis.py`; outputs `results.json` and `figures/comparison.png`.
- Rows with missing text or label are dropped during parsing.

## Reference

This project is motivated by the following paper:

> MAA Anindya, S Biswas, A Iqbal, J Sarker (2026). *ToxiShield: Promoting Inclusive Developer Communication through Real-Time Toxicity Filtering*. Proceedings of the ACM. https://dl.acm.org/doi/abs/10.1145/3808130

Paper author: Prof. Amiangshu Bosu, Wayne State University (amiangshu.bosu@wayne.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
