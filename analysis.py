"""Tuned class-weighted word+char TF-IDF LR (+LightGBM blend, tuned threshold) vs plain word TF-IDF LR
on the ToxiCR code-review toxicity data at ~10% toxic prevalence."""
import io, json, os, sys, time
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier

SEED, SEEDS = 42, [42, 43, 44]
RAW = "https://raw.githubusercontent.com/WSU-SEAL/ToxiCR/"
SRC = "https://github.com/WSU-SEAL/ToxiCR"
np.random.seed(SEED)


def get(url):
    """Download with timeout and backoff on 429/5xx; None on 404."""
    hdr = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.environ.get("GITHUB_TOKEN") and "api." in url else {}
    for i in range(5):
        try:
            r = requests.get(url, timeout=60, headers=hdr)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            if r.status_code not in (429,) and r.status_code < 500:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    return None


def load_raw():
    """Find and read the ToxiCR labelled dataset (xlsx/csv)."""
    urls = [f"{RAW}{b}/{p}" for b in ("master", "main") for p in
            ("dataset/code-review-dataset-full.xlsx", "data/code-review-dataset-full.xlsx",
             "code-review-dataset-full.xlsx", "input/code-review-dataset-full.xlsx")]
    for b in ("master", "main"):  # fallback: discover files through the repo tree
        c = get(f"https://api.github.com/repos/WSU-SEAL/ToxiCR/git/trees/{b}?recursive=1")
        if c:
            paths = [t["path"] for t in json.loads(c)["tree"] if t["path"].lower().endswith((".xlsx", ".csv"))]
            paths.sort(key=lambda p: ("full" not in p.lower(), "code-review" not in p.lower()))
            urls += [f"{RAW}{b}/{p}" for p in paths]
    for u in urls:
        c = get(u)
        if c:
            try:
                df = pd.read_excel(io.BytesIO(c)) if u.endswith(".xlsx") else pd.read_csv(io.BytesIO(c))
            except Exception:
                continue
            print("loaded", u, df.shape)
            return df, u
    sys.exit("could not download dataset")


def parse(df):
    """Pick text and binary toxic-label columns."""
    cols = {c.lower(): c for c in df.columns}
    t = next((cols[k] for k in ("message", "text", "comment", "body") if k in cols), None)
    l = next((cols[k] for k in ("is_toxic", "toxic", "label", "y") if k in cols), None)
    if t is None or l is None:
        sys.exit(f"cannot identify columns in {list(df.columns)}")
    lab = df[l]
    y = lab.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "toxic", "yes"]) if lab.dtype == object else lab.astype(float) > 0
    out = pd.DataFrame({"text": df[t].astype(str), "y": y.astype(int)})
    return out[out.text.str.strip() != ""].drop_duplicates("text").reset_index(drop=True)


def evaluate(y, p, thr):
    """Shared evaluation: toxic F1 at thr, PR-AUC, toxic recall."""
    pred = (p >= thr).astype(int)
    return {"f1": f1_score(y, pred), "pr_auc": average_precision_score(y, p), "recall": recall_score(y, pred)}


def best_thr(y, p):
    """Threshold maximising F1 (used on out-of-fold train predictions only)."""
    ths = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, (p >= t).astype(int)) for t in ths]
    return float(ths[int(np.argmax(f))]), float(max(f))


def feats(ng, tr, others):
    """Word + char_wb TF-IDF fitted on `tr` only."""
    vs = [TfidfVectorizer(ngram_range=ng, sublinear_tf=True, min_df=2, max_features=50000),
          TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=50000)]
    Xtr = hstack([v.fit_transform(tr) for v in vs]).tocsr().astype(np.float32)
    return Xtr, [hstack([v.transform(o) for v in vs]).tocsr().astype(np.float32) for o in others]


def lgbm(seed):
    return LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=31, min_child_samples=10,
                          colsample_bytree=0.3, subsample=0.8, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)


# ---- data: ~10% toxic prevalence, one stratified 80/20 split ----
raw, url = load_raw()
d = parse(raw)
n_raw = len(d)
pos, neg = d[d.y == 1], d[d.y == 0]
if len(pos) < 100 or len(neg) < 100:
    sys.exit("too few examples per class")
n_pos = min(len(pos), int(round(len(neg) / 9)))
n_neg = min(len(neg), 9 * n_pos)
d = pd.concat([pos.sample(n_pos, random_state=SEED), neg.sample(n_neg, random_state=SEED)]).reset_index(drop=True)
Xt, Xe, yt, ye = train_test_split(d.text, d.y.values, test_size=0.2, stratify=d.y.values, random_state=SEED)
Xt, Xe = Xt.tolist(), Xe.tolist()
print(f"rows {len(d)} (clean {n_raw}), prevalence {d.y.mean():.3f}, train {len(Xt)}, test {len(Xe)}")

# ---- baseline: word 1-2gram TF-IDF + default LR, threshold 0.5 ----
base = {k: [] for k in ("f1", "pr_auc", "recall")}
for s in SEEDS:
    v = TfidfVectorizer(ngram_range=(1, 2))
    m = LogisticRegression(max_iter=1000, random_state=s).fit(v.fit_transform(Xt), yt)
    for k, x in evaluate(ye, m.predict_proba(v.transform(Xe))[:, 1], 0.5).items():
        base[k].append(x)
print("baseline done")

# ---- improved: tune C and word n-gram range by 5-fold CV on train; cache fold features ----
skf = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xt, yt))
Xa = np.array(Xt, dtype=object)
cache, oof, grid_C = {}, {}, [0.5, 2, 8]
for ng in [(1, 1), (1, 2)]:
    cache[ng] = []
    for C in grid_C:
        oof[(ng, C)] = np.zeros(len(yt))
    for tr, va in skf:
        A, (B,) = feats(ng, list(Xa[tr]), [list(Xa[va])])
        cache[ng].append((A, B))
        for C in grid_C:
            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED).fit(A, yt[tr])
            oof[(ng, C)][va] = lr.predict_proba(B)[:, 1]
    print("CV done for ngram", ng)
scores = {k: best_thr(yt, p)[1] for k, p in oof.items()}
(ng_b, C_b) = max(scores, key=scores.get)
print("selected", ng_b, C_b, "cv f1 %.3f" % scores[(ng_b, C_b)])

# LightGBM OOF on the selected features, then tune blend weight + threshold by CV
p_lgb = np.zeros(len(yt))
for (tr, va), (A, B) in zip(skf, cache[ng_b]):
    p_lgb[va] = lgbm(SEED).fit(A, yt[tr]).predict_proba(B)[:, 1]
p_lr = oof[(ng_b, C_b)]
best = max(((w,) + best_thr(yt, w * p_lgb + (1 - w) * p_lr) for w in [0, 0.3, 0.5, 0.7]), key=lambda r: r[2])
w_b, thr_b, cv_f1 = best
print(f"blend weight {w_b}, threshold {thr_b:.2f}, cv f1 {cv_f1:.3f}")

# ---- final fit on full train, single test evaluation ----
A, (B,) = feats(ng_b, Xt, [Xe])
pl = LogisticRegression(C=C_b, class_weight="balanced", max_iter=1000, random_state=SEED).fit(A, yt).predict_proba(B)[:, 1]
imp = {k: [] for k in base}
for s in SEEDS:
    pg = lgbm(s).fit(A, yt).predict_proba(B)[:, 1] if w_b > 0 else 0
    for k, x in evaluate(ye, w_b * pg + (1 - w_b) * pl, thr_b).items():
        imp[k].append(x)

M = {}
for k in base:
    M[f"baseline_{k}"], M[f"improved_{k}"] = float(np.mean(base[k])), float(np.mean(imp[k]))
    M[f"baseline_{k}_std"], M[f"improved_{k}_std"] = float(np.std(base[k])), float(np.std(imp[k]))
M["baseline_score"], M["improved_score"] = M["baseline_f1"], M["improved_f1"]
M["baseline_score_std"], M["improved_score_std"] = M["baseline_f1_std"], M["improved_f1_std"]
M["improvement"] = M["improved_score"] - M["baseline_score"]
M.update({"selected_C": float(C_b), "selected_word_ngram_max": float(ng_b[1]), "blend_weight_lgbm": float(w_b),
          "tuned_threshold": thr_b, "cv_f1_improved_train": float(cv_f1), "test_toxic_prevalence": float(np.mean(ye))})
if not all(np.isfinite(v) for v in M.values()) or M["baseline_f1"] <= 0:
    sys.exit("non-meaningful metrics")

os.makedirs("figures", exist_ok=True)
ks = ["f1", "pr_auc", "recall"]
x = np.arange(3)
plt.bar(x - 0.2, [M[f"baseline_{k}"] for k in ks], 0.4, label="baseline")
plt.bar(x + 0.2, [M[f"improved_{k}"] for k in ks], 0.4, label="improved")
plt.xticks(x, ["toxic F1", "PR-AUC", "toxic recall"])
plt.legend(); plt.title("Held-out test set"); plt.tight_layout(); plt.savefig("figures/comparison.png", dpi=120)

res = {
    "question": "Can a CV-tuned, class-weighted word+char TF-IDF linear model with a tuned threshold beat a standard word-TF-IDF logistic regression at detecting toxic code review comments when toxicity is rare?",
    "method": "ToxiCR comments, toxic subsampled to ~10% prevalence, one stratified 80/20 split; improved model tuned with 5-fold CV on train only; test used once.",
    "dataset": {"name": "ToxiCR code review toxicity dataset (stand-in)", "source_url": SRC, "n_rows": int(len(d)),
                "n_train": len(Xt), "n_test": len(Xe), "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "toxic-class F1 (test)", "higher_is_better": True,
    "baseline": "Word 1-2gram TF-IDF + default logistic regression, threshold 0.5.",
    "improved": "Word+char_wb TF-IDF, class-weighted LR with CV-tuned C / word n-gram range / decision threshold, optionally blended with LightGBM (blend weight CV-tuned).",
    "metrics": M,
    "notes": "Stand-in dataset, not the ToxiShield paper's data; baseline is a generic stand-in, not ToxiShield. Single split; "
             "seed std reflects only model randomness (baseline LR is deterministic, LightGBM varies). Subsampling changes prevalence vs the original data. "
             "Dataset URL discovered at run time from the repo; exact file used: " + url,
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(json.dumps(M, indent=2))
