# Toxic code review comment detection (tuned TF-IDF)

Compares a CV-tuned, class-weighted word+char TF-IDF logistic regression (optionally blended with LightGBM, with a CV-tuned decision threshold) against a plain word TF-IDF logistic regression on the ToxiCR code-review toxicity dataset, subsampled to ~10% toxic prevalence.

- Single stratified 80/20 split, seed 42; all tuning uses 5-fold CV on the training set only.
- Metric: toxic-class F1 on the held-out test set (also PR-AUC and toxic recall).
- Run `python analysis.py`; outputs `results.json` and `figures/comparison.png`.
- Rows with missing text or label are dropped during parsing.

## Results (latest run)

On a ToxiCR-derived stand-in dataset (~10% toxic, single 80/20 split), a standard word-TF-IDF logistic-regression baseline reached 24% toxic-class F1 on the test set, while my CV-tuned, class-weighted word+character TF-IDF model with a tuned threshold reached 73%.

| Metric | Value |
|---|---|
| baseline_f1 | 0.2375 |
| improved_f1 | 0.7306 |
| baseline_f1_std | 0 |
| improved_f1_std | 0.004835 |
| baseline_pr_auc | 0.6597 |
| improved_pr_auc | 0.8134 |
| baseline_pr_auc_std | 0 |
| improved_pr_auc_std | 0.001045 |
| baseline_recall | 0.1348 |
| improved_recall | 0.7174 |
| baseline_recall_std | 0 |
| improved_recall_std | 0.0071 |
| baseline_score | 0.2375 |
| improved_score | 0.7306 |
| baseline_score_std | 0 |
| improved_score_std | 0.004835 |
| improvement | 0.4931 |
| selected_C | 8 |
| selected_word_ngram_max | 2 |
| blend_weight_lgbm | 0.7 |
| tuned_threshold | 0.22 |
| cv_f1_improved_train | 0.7327 |
| test_toxic_prevalence | 0.09991 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset, not the ToxiShield paper's data; baseline is a generic stand-in, not ToxiShield. Single split; seed std reflects only model randomness (baseline LR is deterministic, LightGBM varies). Subsampling changes prevalence vs the original data. Rows with missing text or label are dropped before use. Dataset URL discovered at run time from the repo; exact file used: https://raw.githubusercontent.com/WSU-SEAL/ToxiCR/master/models/code-review-dataset-full.xlsx

## Reference

This project is motivated by the following paper:

> MAA Anindya, S Biswas, A Iqbal, J Sarker (2026). *ToxiShield: Promoting Inclusive Developer Communication through Real-Time Toxicity Filtering*. Proceedings of the ACM. https://dl.acm.org/doi/abs/10.1145/3808130

Paper author: Prof. Amiangshu Bosu, Wayne State University (amiangshu.bosu@wayne.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
