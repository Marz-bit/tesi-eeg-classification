from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

CSV_PATH = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9 classes\tree_9class_train_F-1\20260109-113103.csv")
OUT_DIR = CSV_PATH.parent / "stats_cs_vs_tl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_COL = "Subject"
METRICS = [
    ("Unseen Acc.", "Tuned Acc.", "Accuracy"),
    ("Unseen F1",   "Tuned F1",   "Macro-F1"),
]
ALPHA = 0.05

def detect_sep(p: Path) -> str:
    head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "\t" in head: return "\t"
    if ";" in head:  return ";"
    return ","

def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")

def mean_ci_t(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf((1 + conf) / 2.0, df=n-1)
    mu = x.mean()
    return mu - tcrit*se, mu + tcrit*se

def save_both(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))

def p_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2)
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom")

def boxplot_unseen_tuned_with_sig(x, y, metric_name, p_corr, test_used, outbase: Path):
    """
    Two boxplots (Unseen vs Tuned) + significance bracket with stars based on corrected p-value.
    Clean layout: no overlapping text.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.boxplot([x, y], showmeans=True, meanprops=dict(marker="+", markersize=12))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Unseen", "Tuned"])
    ax.set_ylabel(metric_name)

    # Title with extra padding (space from axes)
    ax.set_title(f"Unseen vs Tuned — {metric_name}", pad=18)

    # bracket positioning
    y_max = max(np.max(x), np.max(y))
    y_min = min(np.min(x), np.min(y))
    y_rng = max(y_max - y_min, 1e-6)
    yb = y_max + 0.08 * y_rng
    h = 0.02 * y_rng

    stars = p_to_stars(p_corr)
    add_sig_bracket(ax, 1, 2, yb, h, stars)

    # Put test info INSIDE the axes (top-left), so it never overlaps the title
    ax.text(
        0.02, 0.98,
        f"{test_used}, p(corr)={p_corr:.3g}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9
    )

    fig.tight_layout()
    save_both(fig, outbase)
    plt.close(fig)


sep = detect_sep(CSV_PATH)
df = pd.read_csv(CSV_PATH, sep=sep)
df = df.dropna(subset=[SUBJECT_COL]).copy()

for a, b, _ in METRICS:
    df[a] = to_float(df[a])
    df[b] = to_float(df[b])

df = df.dropna(subset=[c for a, b, _ in METRICS for c in (a, b)]).copy()


subj = df.groupby(SUBJECT_COL)[[c for a, b, _ in METRICS for c in (a, b)]].mean()
subj["n_folds"] = df.groupby(SUBJECT_COL).size()

for a, b, name in METRICS:
    subj[f"delta_{name}"] = subj[b] - subj[a]

# Save subject-level data
subj.reset_index().to_csv(OUT_DIR / "subjectlevel_means_and_deltas.csv", index=False)

rows = []
m_tests = len(METRICS)  # Bonferroni across the two metrics

for a, b, name in METRICS:
    x = subj[a].to_numpy(dtype=float)  # Unseen
    y = subj[b].to_numpy(dtype=float)  # Tuned
    d = subj[f"delta_{name}"].to_numpy(dtype=float)
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
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan

    p_bonf = min(p * m_tests, 1.0)
    ci_low, ci_high = mean_ci_t(d, conf=0.95)

    rows.append({
        "metric": name,
        "n_subjects": n,
        "mean_unseen": float(np.mean(x)),
        "std_unseen": float(np.std(x, ddof=1)),
        "mean_tuned": float(np.mean(y)),
        "std_tuned": float(np.std(y, ddof=1)),
        "mean_delta": float(np.mean(d)),
        "std_delta": float(np.std(d, ddof=1)),
        "ci95_delta_low": float(ci_low),
        "ci95_delta_high": float(ci_high),
        "shapiro_p(delta)": float(shapiro_p),
        "test_used": test_used,
        "statistic": stat,
        "p_raw": p,
        "p_bonferroni": float(p_bonf),
        "cohen_dz": float(dz),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "summary_subjectlevel_unseen_vs_tuned_stats.csv", index=False)


for a, b, name in METRICS:
    tag = name.replace("-", "").replace(" ", "")
    x = subj[a].to_numpy(dtype=float)
    y = subj[b].to_numpy(dtype=float)

    p_corr = float(summary.loc[summary["metric"] == name, "p_bonferroni"].values[0])
    test_used = str(summary.loc[summary["metric"] == name, "test_used"].values[0])

    boxplot_unseen_tuned_with_sig(
        x, y, name,
        p_corr=p_corr,
        test_used=test_used,
        outbase=OUT_DIR / f"box_unseen_vs_tuned_withsig_{tag}"
    )

print("Done. Results saved in:", OUT_DIR)

