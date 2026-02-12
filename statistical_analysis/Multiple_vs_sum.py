from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


CSV_PROD = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9 classes\tree_9class_train_F-1\20260109-113103.csv")  # routing via multiplication
CSV_SUM  = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\Dendro\CM\20260208-233502.csv")      # routing via sum
OUT_DIR = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\statistical analysis\+ vs. x\brackets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_COL = "Subject"
COLMAP = {
    "unseen_acc": "Unseen Acc.",
    "unseen_f1":  "Unseen F1",
    "tuned_acc":  "Tuned Acc.",
    "tuned_f1":   "Tuned F1",
}
ALPHA = 0.05

def detect_sep(p: Path) -> str:
    head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "\t" in head: return "\t"
    if ";" in head:  return ";"
    return ","

def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False),
                         errors="coerce")

def mean_ci_t(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf((1 + conf) / 2.0, df=n-1)
    mu = x.mean()
    return mu - tcrit*se, mu + tcrit*se

def holm_correction(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(m, dtype=float)
    for i, p in enumerate(ranked):
        adj[i] = min((m - i) * p, 1.0)
    for i in range(1, m):
        adj[i] = max(adj[i], adj[i-1])
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out

def p_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c="C0")
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom")

def read_and_aggregate_subject_level(csv_path: Path) -> pd.DataFrame:
    sep = detect_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)
    df = df.dropna(subset=[SUBJECT_COL]).copy()

    for col in COLMAP.values():
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")
        df[col] = to_float(df[col])

    df = df.dropna(subset=list(COLMAP.values())).copy()

    # subject-level mean over folds
    subj = df.groupby(SUBJECT_COL)[list(COLMAP.values())].mean()
    subj["n_folds"] = df.groupby(SUBJECT_COL).size()
    subj = subj.reset_index()
    return subj


prod = read_and_aggregate_subject_level(CSV_PROD).rename(columns={v: k for k, v in COLMAP.items()})
sum_ = read_and_aggregate_subject_level(CSV_SUM ).rename(columns={v: k for k, v in COLMAP.items()})

prod["method"] = "product"
sum_["method"] = "sum"

common_subjects = np.intersect1d(prod[SUBJECT_COL].unique(), sum_[SUBJECT_COL].unique())
prod = prod[prod[SUBJECT_COL].isin(common_subjects)].copy()
sum_ = sum_[sum_[SUBJECT_COL].isin(common_subjects)].copy()

# Merge for easy paired comparisons
merged = prod.merge(sum_, on=SUBJECT_COL, suffixes=("_prod", "_sum"))
merged.to_csv(OUT_DIR / "subjectlevel_paired_values.csv", index=False)

OUTCOMES = [
    ("unseen_acc", "Unseen Accuracy"),
    ("unseen_f1",  "Unseen Macro-F1"),
    ("tuned_acc",  "Tuned Accuracy"),
    ("tuned_f1",   "Tuned Macro-F1"),
]

rows = []
pvals = []


for key, label in OUTCOMES:
    x = merged[f"{key}_prod"].to_numpy(dtype=float)
    y = merged[f"{key}_sum"].to_numpy(dtype=float)
    d = y - x
    n = len(d)

    shapiro_p = stats.shapiro(d).pvalue if n >= 3 else np.nan

    if (not np.isnan(shapiro_p)) and (shapiro_p >= ALPHA):
        test_used = "paired t-test"
        t = stats.ttest_rel(y, x)
        stat = float(t.statistic)
        p = float(t.pvalue)
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
    else:
        test_used = "Wilcoxon signed-rank"
        w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        stat = float(w.statistic)
        p = float(w.pvalue)
        # dz still useful as standardized paired difference
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan

    ci_low, ci_high = mean_ci_t(d, conf=0.95)

    rows.append({
        "outcome": key,
        "label": label,
        "n_subjects": n,
        "mean_prod": float(np.mean(x)),
        "mean_sum": float(np.mean(y)),
        "mean_delta(sum-prod)": float(np.mean(d)),
        "ci95_delta_low": float(ci_low),
        "ci95_delta_high": float(ci_high),
        "shapiro_p(delta)": float(shapiro_p),
        "test_used": test_used,
        "statistic": stat,
        "p_raw": p,
        "cohen_dz": dz,
    })
    pvals.append(p)

# Multiple-comparison correction across the 4 outcomes
p_holm = holm_correction(np.array(pvals))
for r, ph in zip(rows, p_holm):
    r["p_holm_4outcomes"] = float(ph)

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "routing_compare_summary.csv", index=False)

for key, label in OUTCOMES:
    x = merged[f"{key}_prod"].to_numpy(dtype=float)
    y = merged[f"{key}_sum"].to_numpy(dtype=float)

    # corresponding corrected p for stars
    ph = float(summary.loc[summary["outcome"] == key, "p_holm_4outcomes"].values[0])

    fig, ax = plt.subplots(figsize=(7,5))

    ax.boxplot([x, y], showmeans=True, meanprops=dict(marker="+", markersize=12))
    ax.set_xticks([1,2])
    ax.set_xticklabels(["Product routing", "Sum routing"])
    ax.set_ylabel(label)
    ax.set_title(f"{label}: Product vs Sum routing (subject-level)")

    # paired lines + points
    for xi, yi in zip(x, y):
        ax.plot([1,2], [xi, yi], marker="o", alpha=0.6)

    # significance bracket
    y_max = max(np.max(x), np.max(y))
    y_min = min(np.min(x), np.min(y))
    y_range = max(y_max - y_min, 1e-6)
    yb = y_max + 0.08*y_range
    add_sig_bracket(ax, 1, 2, yb, 0.02*y_range, p_to_stars(ph))

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"routing_{key}.png", dpi=300)
    fig.savefig(OUT_DIR / f"routing_{key}.pdf")
    plt.close(fig)

print("Done. Results in:", OUT_DIR)
