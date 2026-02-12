from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from itertools import combinations



METHODS = [
    ("Heatmaps", Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\Dendro\NONE\20260208-152756.csv")),
    ("Last-layer", Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\Dendro\EMB\20260208-235329.csv")),
    ("Softmax", Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\Dendro\SOFT\20260208-163246.csv")),
    ("CM", Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\Dendro\CM\20260208-233502.csv")),
]
BASELINE_NAME = METHODS[0][0]  # annotate vs this
OUT_DIR = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\Dendro")
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

def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (FWER)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(m, dtype=float)
    for i, p in enumerate(ranked):
        adj[i] = min((m - i) * p, 1.0)
    for i in range(1, m):  # enforce monotonicity
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
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2)
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
    subj = df.groupby(SUBJECT_COL)[list(COLMAP.values())].mean().reset_index()
    subj["n_folds"] = df.groupby(SUBJECT_COL).size().values
    return subj

def residuals_for_rm(mat: pd.DataFrame) -> np.ndarray:
    """
    Residuals removing subject and condition means:
    r_ij = y_ij - subj_mean_i - cond_mean_j + grand_mean
    """
    grand = mat.to_numpy().mean()
    subj_mean = mat.mean(axis=1)
    cond_mean = mat.mean(axis=0)
    resid = mat.sub(subj_mean, axis=0).sub(cond_mean, axis=1) + grand
    return resid.to_numpy().ravel()

def gg_epsilon(mat: pd.DataFrame) -> float:
    """
    Greenhouse–Geisser epsilon from covariance of conditions.
    """
    Y = mat.to_numpy()
    S = np.cov(Y, rowvar=False, ddof=1)  # k x k
    k = S.shape[0]
    trS = np.trace(S)
    trSS = np.trace(S @ S)
    eps = (trS**2) / ((k - 1) * trSS) if trSS > 0 else np.nan
    return float(np.clip(eps, 1e-6, 1.0))

def rm_anova_with_gg(mat: pd.DataFrame):
    """
    RM-ANOVA F statistic (sphericity-assuming) computed via classical within-subject ANOVA,
    then apply GG correction to df for a conservative p-value.
    Returns: F, df1, df2, p_gg, eps_gg
    """
    Y = mat.to_numpy()  # n x k
    n, k = Y.shape
    grand = Y.mean()
    subj_means = Y.mean(axis=1, keepdims=True)
    cond_means = Y.mean(axis=0, keepdims=True)

    # Sums of squares
    ss_total = np.sum((Y - grand)**2)
    ss_subject = k * np.sum((subj_means - grand)**2)
    ss_condition = n * np.sum((cond_means - grand)**2)
    ss_error = ss_total - ss_subject - ss_condition

    df1 = k - 1
    df2 = (n - 1) * (k - 1)

    ms_condition = ss_condition / df1
    ms_error = ss_error / df2
    F = ms_condition / ms_error

    eps = gg_epsilon(mat)
    df1_gg = df1 * eps
    df2_gg = df2 * eps
    p_gg = stats.f.sf(F, df1_gg, df2_gg)
    return float(F), float(df1_gg), float(df2_gg), float(p_gg), float(eps)

rows = []
methods_order = [m for m, _ in METHODS]

for method_name, path in METHODS:
    subj = read_and_aggregate_subject_level(path)
    for _, r in subj.iterrows():
        rows.append({
            "Subject": r[SUBJECT_COL],
            "method": method_name,
            "unseen_acc": r[COLMAP["unseen_acc"]],
            "unseen_f1":  r[COLMAP["unseen_f1"]],
            "tuned_acc":  r[COLMAP["tuned_acc"]],
            "tuned_f1":   r[COLMAP["tuned_f1"]],
        })

data = pd.DataFrame(rows)

# Keep only subjects present in ALL methods (paired complete-case)
subj_counts = data.groupby("Subject")["method"].nunique()
common_subjects = subj_counts[subj_counts == len(methods_order)].index
data = data[data["Subject"].isin(common_subjects)].copy()

data.to_csv(OUT_DIR / "subjectlevel_all_methods_long.csv", index=False)

OUTCOMES = [
    ("unseen_acc", "Unseen Accuracy"),
    ("unseen_f1",  "Unseen Macro-F1"),
    ("tuned_acc",  "Tuned Accuracy"),
    ("tuned_f1",   "Tuned Macro-F1"),
]

omnibus = []
posthoc = []

for col, label in OUTCOMES:
    mat = data.pivot(index="Subject", columns="method", values=col)
    mat = mat[methods_order].dropna()
    n = mat.shape[0]
    k = mat.shape[1]

    # Normality check on RM residuals
    resid = residuals_for_rm(mat)
    shapiro_p = stats.shapiro(resid).pvalue
    skew = stats.skew(resid, bias=False)

    if shapiro_p >= ALPHA:
        # Parametric path: RM-ANOVA with GG-corrected p (conservative)
        F, df1_gg, df2_gg, p_gg, eps = rm_anova_with_gg(mat)
        omnibus.append({
            "outcome": col, "label": label, "n_subjects": n, "k_conditions": k,
            "resid_shapiro_p": float(shapiro_p), "resid_skewness": float(skew),
            "test": "RM-ANOVA (GG-corrected)",
            "F": F, "df1_GG": df1_gg, "df2_GG": df2_gg, "epsilon_GG": eps,
            "p_GG": p_gg
        })
        parametric = True
    else:
        # Nonparametric path: Friedman
        arrays = [mat[m].to_numpy() for m in methods_order]
        chi2, p_fr = stats.friedmanchisquare(*arrays)
        omnibus.append({
            "outcome": col, "label": label, "n_subjects": n, "k_conditions": k,
            "resid_shapiro_p": float(shapiro_p), "resid_skewness": float(skew),
            "test": "Friedman",
            "chi2": float(chi2), "p_raw": float(p_fr)
        })
        parametric = False

    # Post-hoc: all pairwise comparisons among methods (6 comparisons), Holm correction
    pairs = list(combinations(methods_order, 2))
    pvals, stats_list, dz_list = [], [], []

    test_name = "paired t-test" if parametric else "Wilcoxon signed-rank"

    for m1, m2 in pairs:
        a = mat[m1].to_numpy()
        b = mat[m2].to_numpy()
        d = b - a  # (m2 - m1)

        if parametric:
            t = stats.ttest_rel(b, a)
            stat = float(t.statistic)
            p = float(t.pvalue)
        else:
            w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            stat = float(w.statistic)
            p = float(w.pvalue)

        sd = np.std(d, ddof=1)
        dz = float(np.mean(d) / sd) if sd > 0 else np.nan

        pvals.append(p)
        stats_list.append(stat)
        dz_list.append(dz)

    p_holm = holm_correction(np.array(pvals))

    for (m1, m2), stat, p, ph, dz in zip(pairs, stats_list, pvals, p_holm, dz_list):
        posthoc.append({
            "outcome": col, "label": label,
            "method_a": m1, "method_b": m2,
            "test": test_name,
            "statistic": stat,
            "p_raw": p,
            "p_holm": ph,
            "cohen_dz_on_diff_(b-a)": dz,
        })

    # Plot (ONCE per metric)
    values = [mat[m].to_numpy() for m in methods_order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(values, showmeans=True, meanprops=dict(marker="+", markersize=12))
    ax.set_xticks(range(1, len(methods_order) + 1))
    ax.set_xticklabels(methods_order, rotation=20, ha="right")
    ax.set_title(f"{label} across hierarchical splits (subject-level)")
    ax.set_ylabel(label)

    y_max = np.max([np.max(v) for v in values])
    y_min = np.min([np.min(v) for v in values])
    y_range = max(y_max - y_min, 1e-6)
    y = y_max + 0.08 * y_range
    h = 0.02 * y_range
    step = 0.07 * y_range

    for (m1, m2), ph in zip(pairs, p_holm):
        if ph < ALPHA:
            x1 = methods_order.index(m1) + 1
            x2 = methods_order.index(m2) + 1
            add_sig_bracket(ax, x1, x2, y, h, p_to_stars(ph))
            y += step

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"box_{col}.png", dpi=300)
    fig.savefig(OUT_DIR / f"box_{col}.pdf")
    plt.close(fig)

# Save stats
pd.DataFrame(omnibus).to_csv(OUT_DIR / "omnibus_results_auto_param_vs_friedman.csv", index=False)
pd.DataFrame(posthoc).to_csv(OUT_DIR / "posthoc_pairwise_holm.csv", index=False)

print("Done. Saved to:", OUT_DIR)
