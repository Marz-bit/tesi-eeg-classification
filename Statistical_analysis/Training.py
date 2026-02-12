from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATHS = {
    "TL-only": Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\Dendro\CM\20260208-233502.csv"),
    "CS": Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9_TRAINING\F-1\9_hybrid_iniz_F-1\20260127-155132.csv"),
    "From-scratch":        Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9_TRAINING\F-1\9_hybrid_F-1\20260127-155533.csv"),
}

OUT_DIR = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\statistical analysis\Training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_COL = "Subject"

# These must match your CSV headers (edit if needed)
COLMAP = {
    "Unseen Accuracy": "Unseen Acc.",
    "Unseen Macro-F1": "Unseen F1",
    "Tuned Accuracy":  "Tuned Acc.",
    "Tuned Macro-F1":  "Tuned F1",
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

def holm_correction(pvals):
    """
    Holm step-down: returns adjusted p-values in original order.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)

    running_max = 0.0
    for i, idx in enumerate(order):
        p = pvals[idx]
        adj_p = (m - i) * p
        running_max = max(running_max, adj_p)
        adj[idx] = min(running_max, 1.0)
    return adj

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def rm_residuals_matrix(Y):
    """
    Y: (n_subjects, k_conditions)
    Returns residuals: Y - subj_mean - cond_mean + grand_mean
    """
    subj_mean = Y.mean(axis=1, keepdims=True)
    cond_mean = Y.mean(axis=0, keepdims=True)
    grand = Y.mean()
    return Y - subj_mean - cond_mean + grand

def rm_anova_oneway(Y):
    """
    Classic one-way repeated-measures ANOVA (within-subject factor).
    Returns: F, p_uncorr, p_GG, eps_GG, df1, df2, df1_GG, df2_GG
    """
    Y = np.asarray(Y, dtype=float)
    n, k = Y.shape
    grand = Y.mean()
    subj_means = Y.mean(axis=1)
    cond_means = Y.mean(axis=0)

    ss_total = np.sum((Y - grand) ** 2)
    ss_subj = k * np.sum((subj_means - grand) ** 2)
    ss_cond = n * np.sum((cond_means - grand) ** 2)
    ss_err = ss_total - ss_subj - ss_cond

    df1 = k - 1
    df2 = (n - 1) * (k - 1)

    ms_cond = ss_cond / df1
    ms_err = ss_err / df2
    F = ms_cond / ms_err if ms_err > 0 else np.nan
    p_unc = 1.0 - stats.f.cdf(F, df1, df2) if np.isfinite(F) else np.nan

    # Greenhouse–Geisser epsilon
    Yc = Y - Y.mean(axis=1, keepdims=True)  # remove subject mean
    S = np.cov(Yc, rowvar=False, ddof=1)    # k x k covariance
    trS = np.trace(S)
    trS2 = np.trace(S @ S)
    eps = (trS ** 2) / ((k - 1) * trS2) if trS2 > 0 else 1.0
    eps = float(np.clip(eps, 1.0 / (k - 1), 1.0))

    df1_gg = eps * df1
    df2_gg = eps * df2
    p_gg = 1.0 - stats.f.cdf(F, df1_gg, df2_gg) if np.isfinite(F) else np.nan

    return float(F), float(p_unc), float(p_gg), float(eps), float(df1), float(df2), float(df1_gg), float(df2_gg)

def add_sig_bracket(ax, x1, x2, y, h, text, lw=1.5):
    """
    Draw bracket between x1 and x2 at height y, with small height h.
    x positions are 1-based boxplot positions.
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=lw)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom")

def boxplot_with_sig(ax, data_by_method, labels, ylabel, title, sig_pairs):
    """
    data_by_method: list of arrays, one per method, aligned by subject
    sig_pairs: list of tuples (i, j, p_adj) where i<j are 0-based method indices,
               include ONLY significant ones.
    """
    ax.boxplot(
        data_by_method,
        labels=labels,
        showmeans=True,
        meanprops=dict(marker="+", markersize=12),
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # annotate only significant comparisons, stack brackets nicely
    if sig_pairs:
        ymax = max(np.nanmax(d) for d in data_by_method)
        ymin = min(np.nanmin(d) for d in data_by_method)
        yr = ymax - ymin if ymax > ymin else 1.0
        base = ymax + 0.05 * yr
        step = 0.06 * yr
        h = 0.02 * yr

        # sort by span (shorter first) to reduce clutter
        sig_pairs = sorted(sig_pairs, key=lambda t: (t[1]-t[0], t[0], t[1]))

        for s, (i, j, p) in enumerate(sig_pairs):
            y = base + s * step
            add_sig_bracket(ax, i+1, j+1, y, h, stars(p))

        ax.set_ylim(ymin - 0.05*yr, base + (len(sig_pairs)+1)*step)


method_tables = []
for method, path in CSV_PATHS.items():
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep)

    if SUBJECT_COL not in df.columns:
        raise ValueError(f"{path.name}: missing subject column '{SUBJECT_COL}'")

    # keep needed columns
    needed_cols = [SUBJECT_COL] + list(COLMAP.values())
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    df = df.dropna(subset=[SUBJECT_COL]).copy()
    for out_name, col in COLMAP.items():
        df[col] = to_float(df[col])

    df = df.dropna(subset=list(COLMAP.values())).copy()

    # subject-level mean across folds
    subj = df.groupby(SUBJECT_COL)[list(COLMAP.values())].mean()
    subj["n_folds"] = df.groupby(SUBJECT_COL).size()
    subj["Method"] = method

    subj = subj.reset_index()
    # rename columns to unified outcome names
    ren = {v: k for k, v in COLMAP.items()}
    subj = subj.rename(columns=ren)

    method_tables.append(subj)

all_long = pd.concat(method_tables, ignore_index=True)

# keep only subjects present in ALL methods (paired repeated measures)
methods = list(CSV_PATHS.keys())
common_subjects = set(all_long[all_long["Method"] == methods[0]][SUBJECT_COL])
for m in methods[1:]:
    common_subjects &= set(all_long[all_long["Method"] == m][SUBJECT_COL])
common_subjects = sorted(common_subjects)

all_long = all_long[all_long[SUBJECT_COL].isin(common_subjects)].copy()
all_long.to_csv(OUT_DIR / "subjectlevel_long.csv", index=False)


outcomes = list(COLMAP.keys())  # 4 outcomes
omnibus_rows = []
posthoc_rows = []

# build wide matrices per outcome: rows=subjects, cols=methods
wide = {}
for outcome in outcomes:
    piv = all_long.pivot(index=SUBJECT_COL, columns="Method", values=outcome)
    piv = piv.loc[common_subjects, methods]  # enforce ordering
    wide[outcome] = piv

# Omnibus tests (with Holm across the 4 outcomes)
omnibus_p_raw = []

tmp_results = {}  # store per-outcome decisions for later
for outcome in outcomes:
    Y = wide[outcome].to_numpy()  # (n,k)
    n, k = Y.shape

    resid = rm_residuals_matrix(Y).ravel()
    sh_p = stats.shapiro(resid).pvalue if resid.size >= 3 else np.nan

    if (not np.isnan(sh_p)) and (sh_p >= ALPHA):
        path = "RM-ANOVA (GG)"
        F, p_unc, p_gg, eps, df1, df2, df1_gg, df2_gg = rm_anova_oneway(Y)
        stat = F
        p_raw = p_gg  # use GG-corrected p as primary
        extra = dict(epsilon_GG=eps, df1_GG=df1_gg, df2_GG=df2_gg, p_uncorrected=p_unc)
    else:
        path = "Friedman"
        cols = [Y[:, j] for j in range(Y.shape[1])]
        fr = stats.friedmanchisquare(*cols)
        stat = float(fr.statistic)
        p_raw = float(fr.pvalue)
        extra = dict(epsilon_GG=np.nan, df1_GG=np.nan, df2_GG=np.nan, p_uncorrected=np.nan)

    omnibus_p_raw.append(p_raw)
    tmp_results[outcome] = dict(
        outcome=outcome, n_subjects=n, k_methods=k,
        shapiro_p_resid=float(sh_p), omnibus_test=path,
        statistic=float(stat), p_raw=float(p_raw),
        **extra
    )

# Holm correction across 4 omnibus tests
omni_p_adj = holm_correction(omnibus_p_raw)

for outcome, p_adj in zip(outcomes, omni_p_adj):
    tmp_results[outcome]["p_holm_across_outcomes"] = float(p_adj)
    omnibus_rows.append(tmp_results[outcome])

omnibus_df = pd.DataFrame(omnibus_rows)
omnibus_df.to_csv(OUT_DIR / "omnibus_results.csv", index=False)

# Post-hoc only when omnibus is significant AFTER Holm-across-outcomes
pairs = list(itertools.combinations(range(len(methods)), 2))  # 3 pairs for k=3

for outcome in outcomes:
    res = tmp_results[outcome]
    if res["p_holm_across_outcomes"] >= ALPHA:
        continue

    Y = wide[outcome].to_numpy()
    # choose post-hoc family based on omnibus path:
    # - if RM-ANOVA path: paired t-tests
    # - if Friedman path: Wilcoxon
    use_param = (res["omnibus_test"] == "RM-ANOVA (GG)")

    pvals = []
    tmp_pair = []
    for i, j in pairs:
        a = Y[:, i]
        b = Y[:, j]
        d = b - a

        if use_param:
            t = stats.ttest_rel(b, a)
            stat = float(t.statistic)
            p = float(t.pvalue)
        else:
            w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            stat = float(w.statistic)
            p = float(w.pvalue)

        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan

        pvals.append(p)
        tmp_pair.append((i, j, stat, p, dz))

    p_adj = holm_correction(pvals)

    for (i, j, stat, p, dz), padj in zip(tmp_pair, p_adj):
        posthoc_rows.append({
            "outcome": outcome,
            "method_A": methods[i],
            "method_B": methods[j],
            "test_used": "paired t-test" if use_param else "Wilcoxon signed-rank",
            "statistic": stat,
            "p_raw": p,
            "p_holm_within_outcome": float(padj),
            "cohen_dz_on_diff(B-A)": float(dz),
        })

posthoc_df = pd.DataFrame(posthoc_rows)
posthoc_df.to_csv(OUT_DIR / "posthoc_pairwise_holm.csv", index=False)

# build a dict outcome -> list of significant pairs (i,j,p_adj)
sig_map = {o: [] for o in outcomes}
if not posthoc_df.empty:
    for _, r in posthoc_df.iterrows():
        if r["p_holm_within_outcome"] < ALPHA:
            i = methods.index(r["method_A"])
            j = methods.index(r["method_B"])
            # NOTE: bracket expects i<j based on positions
            ii, jj = (i, j) if i < j else (j, i)
            sig_map[r["outcome"]].append((ii, jj, float(r["p_holm_within_outcome"])))

for outcome in outcomes:
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    Y = wide[outcome].to_numpy()
    data = [Y[:, j] for j in range(Y.shape[1])]

    boxplot_with_sig(
        ax=ax,
        data_by_method=data,
        labels=methods,
        ylabel=outcome,
        title=f"{outcome} across training regimes (subject-level)",
        sig_pairs=sig_map[outcome]
    )

    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    tag = outcome.replace(" ", "_").replace("-", "").lower()
    fig.savefig(OUT_DIR / f"box_{tag}_training_regimes.png", dpi=300)
    fig.savefig(OUT_DIR / f"box_{tag}_training_regimes.pdf")
    plt.close(fig)

print("Done. Outputs saved in:", OUT_DIR)
