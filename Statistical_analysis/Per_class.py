from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


CSV_FLAT = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9 classes\9_FLAT\20260109-160943.csv")
CSV_BEST = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9_TRAINING\F-1\9_hybrid_iniz_F-1\20260127-155132.csv")

OUT_DIR = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\statistical analysis\Flat\Per class")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_COL = "Subject"
ALPHA = 0.05
APPLY_HOLM = True  # False => p non corretti


def detect_sep(p: Path) -> str:
    head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "\t" in head: return "\t"
    if ";" in head:  return ";"
    return ","

def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    )

def holm_adjust(pvals):
    """Holm step-down adjustment. Returns adjusted p-values in original order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        factor = (m - i)
        val = factor * ranked[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)

    out = np.empty(m, dtype=float)
    out[order] = adj
    return out

def p_to_stars(p):
    p = float(np.asarray(p).ravel()[0])  # robust: garantisce scalare
    if np.isnan(p): return "n.s."
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "n.s."

def add_sig_bracket(ax, x1, x2, y, h, stars, pval, color="C0"):
    """Aggiunge bracket di significatività con asterisco sopra e p-value accanto."""
    # Disegna la bracket
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color=color)
    
    # Asterisco centrato sopra la bracket (nero)
    ax.text((x1+x2)/2, y+h, stars, ha="center", va="bottom", 
            fontsize=12, fontweight="bold", color="black")
    
    # P-value formattato a destra dell'asterisco (nero)
    if pval < 0.001:
        pval_text = f"p < 0.001"
    else:
        pval_text = f"p = {pval:.3f}"
    
    ax.text((x1+x2)/2 + 0.15, y+h, pval_text, ha="left", va="bottom", 
            fontsize=8, style="italic", color="black")

def mean_ci_t(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    mu = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf((1 + conf) / 2.0, df=n - 1)
    return mu - tcrit * se, mu + tcrit * se

def load_subject_level_tuned_perclass(csv_path: Path) -> pd.DataFrame:
    sep = detect_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)
    df = df.dropna(subset=[SUBJECT_COL]).copy()

    tuned_class_cols = [
        c for c in df.columns
        if c.startswith("Tuned ")
        and c not in ["Tuned Acc.", "Tuned F1"]
    ]
    if len(tuned_class_cols) == 0:
        raise ValueError(
            f"No tuned per-class columns found in {csv_path.name}. "
            f"Expected columns like 'Tuned Left Hand', 'Tuned Word', etc."
        )

    for c in tuned_class_cols:
        df[c] = to_float(df[c])

    df = df.dropna(subset=tuned_class_cols).copy()

    subj = df.groupby(SUBJECT_COL)[tuned_class_cols].mean()
    subj["n_folds"] = df.groupby(SUBJECT_COL).size()
    return subj

flat_subj = load_subject_level_tuned_perclass(CSV_FLAT)
best_subj = load_subject_level_tuned_perclass(CSV_BEST)

# align paired subjects
common_subjects = flat_subj.index.intersection(best_subj.index)
flat_subj = flat_subj.loc[common_subjects].sort_index()
best_subj = best_subj.loc[common_subjects].sort_index()

if len(common_subjects) < 3:
    raise RuntimeError("Not enough common subjects to run paired statistics (need >= 3).")

# common tuned per-class columns
tuned_cols = [c for c in flat_subj.columns if c.startswith("Tuned ") and c not in ["Tuned Acc.", "Tuned F1"]]
tuned_cols = [c for c in tuned_cols if c in best_subj.columns]
tuned_cols = sorted(tuned_cols, key=lambda x: x.replace("Tuned ", ""))

class_names = [c.replace("Tuned ", "") for c in tuned_cols]

rows = []
raw_pvals = []

for c in tuned_cols:
    x = flat_subj[c].to_numpy(dtype=float)
    y = best_subj[c].to_numpy(dtype=float)
    d = y - x
    n = len(d)

    if np.nanstd(d, ddof=1) == 0 or np.allclose(d, d[0], equal_nan=True):
        shapiro_p = np.nan
        test_used = "degenerate (zero-variance deltas)"
        stat = np.nan
        p_raw = 1.0
    else:
        shapiro_p = stats.shapiro(d).pvalue if n >= 3 else np.nan

        if (not np.isnan(shapiro_p)) and (shapiro_p >= ALPHA):
            test_used = "paired t-test"
            res = stats.ttest_rel(y, x)
            stat = float(res.statistic)
            p_raw = float(res.pvalue)
        else:
            test_used = "Wilcoxon signed-rank"
            res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            stat = float(res.statistic)
            p_raw = float(res.pvalue)

    dz = float(np.nanmean(d) / np.nanstd(d, ddof=1)) if np.nanstd(d, ddof=1) > 0 else np.nan
    ci_low, ci_high = mean_ci_t(d, conf=0.95)

    raw_pvals.append(p_raw)

    rows.append({
        "class": c.replace("Tuned ", ""),
        "n_subjects": int(n),
        "mean_flat": float(np.nanmean(x)),
        "mean_best": float(np.nanmean(y)),
        "mean_delta(best-flat)": float(np.nanmean(d)),
        "ci95_delta_low": float(ci_low),
        "ci95_delta_high": float(ci_high),
        "shapiro_p(delta)": float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
        "test_used": test_used,
        "statistic": stat,
        "p_raw": p_raw,
        "cohen_dz": dz,
    })

# Holm correction across classes
if APPLY_HOLM:
    p_corr = holm_adjust(raw_pvals)
else:
    p_corr = np.asarray(raw_pvals, dtype=float)

for r, pc in zip(rows, p_corr):
    r["p_corr"] = float(np.asarray(pc).ravel()[0])
    r["corr_method"] = "Holm" if APPLY_HOLM else "none"

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "summary_tuned_perclass_flat_vs_best.csv", index=False)
print("Saved summary:", OUT_DIR / "summary_tuned_perclass_flat_vs_best.csv")


data_flat = [flat_subj[c].to_numpy(float) for c in tuned_cols]
data_best = [best_subj[c].to_numpy(float) for c in tuned_cols]

n_classes = len(tuned_cols)
base = np.arange(n_classes)
width = 0.36
pos_flat = base - width/2
pos_best = base + width/2

fig, ax = plt.subplots(figsize=(14, 5))

bp_flat = ax.boxplot(
    data_flat, positions=pos_flat, widths=width*0.9,
    patch_artist=True, showmeans=True, showfliers=False,  # True se vuoi i pallini degli outlier
    meanprops=dict(marker="+", markersize=10)
)
bp_best = ax.boxplot(
    data_best, positions=pos_best, widths=width*0.9,
    patch_artist=True, showmeans=True, showfliers=False,
    meanprops=dict(marker="+", markersize=10)
)

# style: hatching
for b in bp_flat["boxes"]:
    b.set(facecolor="white", edgecolor="black", hatch="//")
for b in bp_best["boxes"]:
    b.set(facecolor="white", edgecolor="black", hatch="")

for key in ["whiskers", "caps", "medians", "means"]:
    for item in bp_flat[key] + bp_best[key]:
        item.set(color="black")

ax.set_xticks(base)
ax.set_xticklabels(class_names, rotation=20, ha="right")
ax.set_ylabel("Tuned class-wise Accuracy")
ax.set_title("Tuned accuracy per class: Flat EEGNet vs Best framework (subject-level)")

# significance brackets per class (p_corr se APPLY_HOLM, altrimenti p_raw)
allv = np.concatenate([np.concatenate(data_flat), np.concatenate(data_best)])
ymin, ymax = np.nanmin(allv), np.nanmax(allv)
yr = (ymax - ymin) if ymax > ymin else 1.0

for i in range(n_classes):
    p_val = float(np.asarray(p_corr[i]).ravel()[0])
    stars = p_to_stars(p_val)

    if stars != "n.s.":
        y_i = max(np.nanmax(data_flat[i]), np.nanmax(data_best[i])) + 0.05 * yr
        h_i = 0.04 * yr  # Leggermente più alto per far stare il p-value
        add_sig_bracket(ax, pos_flat[i], pos_best[i], y_i, h_i, stars, p_val, color="C0")

ax.legend(handles=[
    Patch(facecolor="white", edgecolor="black", hatch="//", label="Flat EEGNet"),
    Patch(facecolor="white", edgecolor="black", hatch="", label="Best framework")
], loc="upper left")

ax.set_ylim(top=ymax + 0.18*yr)
plt.tight_layout()

fig.savefig(OUT_DIR / "box_tuned_perclass_flat_vs_best.png", dpi=300)
fig.savefig(OUT_DIR / "box_tuned_perclass_flat_vs_best.pdf")
plt.close(fig)

print("Saved figure:", OUT_DIR / "box_tuned_perclass_flat_vs_best.png")