# Tuned word+char TF-IDF toxicity detector vs plain TF-IDF logistic regression

## Motivation
"ToxiShield: Promoting Inclusive Developer Communication through Real-Time Toxicity Filtering" (2026) describes a tool that filters toxic developer communication during code review. In practice toxic comments are rare, so a detector must stay precise and still catch them. This experiment checks whether careful tuning of a simple linear pipeline helps in that setting.

## Question
Can a cross-validation-tuned, class-weighted word+character TF-IDF linear model with a tuned decision threshold beat a standard word-TF-IDF logistic regression at detecting toxic code review comments when toxic comments are rare?

## Dataset
The public ToxiCR labelled code review comment dataset (WSU-SEAL/ToxiCR on GitHub, downloaded at run time). It is **not** the paper's own data, which is not stated; it is a stand-in. Toxic comments are subsampled to about 10% prevalence, then split once into stratified 80/20 train/test sets.

## Baseline vs improved
- **Baseline:** word unigram/bigram TF-IDF with a default logistic regression and a 0.5 threshold. It is a generic stand-in, not ToxiShield.
- **Improved:** word + character TF-IDF with a class-weighted logistic regression. C, the word n-gram range and the decision threshold are tuned by 5-fold CV on the training split only. The model is optionally blended with a LightGBM on the same features, with the blend weight also chosen by CV.

Main metric: toxic-class F1 on the held-out test set. Secondary metrics: PR-AUC and toxic recall. The test set is used once, at the end, for both models.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> MAA Anindya, S Biswas, A Iqbal, J Sarker (2026). *ToxiShield: Promoting Inclusive Developer Communication through Real-Time Toxicity Filtering*. Proceedings of the ACM. https://dl.acm.org/doi/abs/10.1145/3808130

Paper author: Prof. Amiangshu Bosu, Wayne State University (amiangshu.bosu@wayne.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
