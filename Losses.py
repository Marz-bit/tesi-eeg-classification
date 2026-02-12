from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATHS = {
    "Masked": Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9 classes\tree_9class_train_F-1\20260109-113103.csv"),
    "Leaf":   Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9 classes\tree_9class_LEAFLOSS\20260113-215238.csv"),
    "Hybrid": Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\Dendro\CM\20260208-233502.csv"),
}

OUT_DIR = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\statistical analysis\Losses")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_COL = "Subject"
TUNED_ACC_COL = "Tuned Acc."
TUNED_F1_COL  = "Tuned F1"

ALPHA = 0.05

# =========================
# I/O helpers
# =========================
def detect_sep(p: Path) -> str:
    head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "\t" in head: return "\t"
    if ";" in head:  return ";"
    return ","

def to_float(s: pd.Series) -> pd.Series:
    # handles decimal comma
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    )

def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, p in enumerate(ranked):
        a = (m - i) * p
        a = max(a, prev)          # enforce monotonicity
        adj[i] = min(a, 1.0)
        prev = adj[i]

    out = np.empty(m, dtype=float)
    out[order] = adj
    return out

def stars(p):
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

# =========================
# Stats helpers
# =========================
def rm_residuals(X: np.ndarray) -> np.ndarray:
    """
    Repeated-measures residuals:
    r_ij = x_ij - subj_mean_i - cond_mean_j + grand_mean
    """
    subj_mean = X.mean(axis=1, keepdims=True)
    cond_mean = X.mean(axis=0, keepdims=True)
    grand = X.mean()
    R = X - subj_mean - cond_mean + grand
    return R.ravel()

def rm_anova_oneway(X: np.ndarray):
    """
    One-way repeated-measures ANOVA (manual), plus Greenhouse–Geisser correction.
    X: shape (n_subjects, k_conditions)
    Returns dict with F, df1, df2, p_uncorrected, epsilon_GG, p_GG, eta_p2
    """
    n, k = X.shape
    grand = X.mean()
    subj_mean = X.mean(axis=1)
    cond_mean = X.mean(axis=0)

    ss_total = ((X - grand) ** 2).sum()
    ss_subj  = k * ((subj_mean - grand) ** 2).sum()
    ss_cond  = n * ((cond_mean - grand) ** 2).sum()
    ss_err   = ss_total - ss_subj - ss_cond

    df1 = k - 1
    df2 = (n - 1) * (k - 1)

    ms_cond = ss_cond / df1
    ms_err  = ss_err  / df2
    F = ms_cond / ms_err
    p_unc = stats.f.sf(F, df1, df2)

    # Greenhouse–Geisser epsilon from covariance of conditions
    S = np.cov(X, rowvar=False, ddof=1)  # k x k
    trS = np.trace(S)
    trSS = np.trace(S @ S)
    eps = (trS ** 2) / ((k - 1) * trSS) if trSS > 0 else 1.0
    eps = float(np.clip(eps, 1.0 / (k - 1), 1.0))

    df1_gg = eps * df1
    df2_gg = eps * df2
    p_gg = stats.f.sf(F, df1_gg, df2_gg)

    eta_p2 = ss_cond / (ss_cond + ss_err) if (ss_cond + ss_err) > 0 else np.nan

    return {
        "F": float(F),
        "df1": float(df1),
        "df2": float(df2),
        "p_uncorrected": float(p_unc),
        "epsilon_GG": float(eps),
        "p_GG": float(p_gg),
        "eta_p2": float(eta_p2),
    }

def friedman_test(X: np.ndarray):
    """
    Friedman test + Kendall's W.
    """
    n, k = X.shape
    args = [X[:, j] for j in range(k)]
    res = stats.friedmanchisquare(*args)
    chi2 = float(res.statistic)
    p = float(res.pvalue)
    W = chi2 / (n * (k - 1)) if n * (k - 1) > 0 else np.nan
    return {"chi2": chi2, "df": k - 1, "p": p, "kendalls_W": float(W)}

def paired_effect_dz(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan

def mean_ci_t(d: np.ndarray, conf=0.95):
    d = np.asarray(d, dtype=float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf((1 + conf) / 2.0, df=n-1)
    mu = d.mean()
    return float(mu - tcrit*se), float(mu + tcrit*se)

# =========================
# Plot helper
# =========================
def add_sig_bracket(ax, x1, x2, y, h, text, lw=1.5):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c="C0")
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom")

def boxplot_with_sig(ax, data_list, labels, title, ylabel, sig_pairs):
    # data_list: list of arrays, one per condition, subject-level
    ax.boxplot(data_list, labels=labels, showmeans=True, meanprops=dict(marker="+", markersize=12))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)

    # annotate only significant pairs
    ymax = max([np.max(d) for d in data_list if len(d) > 0])
    ymin = min([np.min(d) for d in data_list if len(d) > 0])
    yrange = ymax - ymin if ymax > ymin else 1.0

    y = ymax + 0.05 * yrange
    h = 0.02 * yrange

    for (i, j, p_corr) in sig_pairs:
        if p_corr < ALPHA:
            add_sig_bracket(ax, i+1, j+1, y, h, stars(p_corr))
            y += 0.06 * yrange  # stack brackets

# =========================
# Load + subject-level aggregation
# =========================
rows_long = []
subj_tables = {}

for loss_name, csv_path in CSV_PATHS.items():
    sep = detect_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)

    # ensure columns exist
    for col in [SUBJECT_COL, TUNED_ACC_COL, TUNED_F1_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    df = df.dropna(subset=[SUBJECT_COL]).copy()
    df[TUNED_ACC_COL] = to_float(df[TUNED_ACC_COL])
    df[TUNED_F1_COL]  = to_float(df[TUNED_F1_COL])
    df = df.dropna(subset=[TUNED_ACC_COL, TUNED_F1_COL]).copy()

    # aggregate folds -> subject
    subj = df.groupby(SUBJECT_COL)[[TUNED_ACC_COL, TUNED_F1_COL]].mean()
    subj_tables[loss_name] = subj

    tmp = subj.reset_index().copy()
    tmp["Loss"] = loss_name
    tmp = tmp.rename(columns={TUNED_ACC_COL: "TunedAccuracy", TUNED_F1_COL: "TunedMacroF1"})
    rows_long.append(tmp)

long_df = pd.concat(rows_long, ignore_index=True)

# pivot to matrices (subjects x loss)
loss_order = list(CSV_PATHS.keys())

wide_acc = long_df.pivot_table(index=SUBJECT_COL, columns="Loss", values="TunedAccuracy")
wide_f1  = long_df.pivot_table(index=SUBJECT_COL, columns="Loss", values="TunedMacroF1")

# keep only subjects present in all losses
wide_acc = wide_acc[loss_order].dropna(axis=0, how="any")
wide_f1  = wide_f1[loss_order].dropna(axis=0, how="any")

# align subjects across metrics
common_subjects = wide_acc.index.intersection(wide_f1.index)
wide_acc = wide_acc.loc[common_subjects]
wide_f1  = wide_f1.loc[common_subjects]

# save subject-level long table
long_df = long_df[long_df[SUBJECT_COL].isin(common_subjects)]
long_df.to_csv(OUT_DIR / "subjectlevel_loss_long.csv", index=False)

# =========================
# Omnibus + posthoc per outcome
# =========================
def analyze_outcome(wide: pd.DataFrame, outcome_name: str):
    X = wide.to_numpy()  # (n_subjects, k=3)
    n, k = X.shape

    # normality on RM residuals
    resid = rm_residuals(X)
    shapiro_p = stats.shapiro(resid).pvalue if len(resid) >= 3 else np.nan

    results = {
        "outcome": outcome_name,
        "n_subjects": int(n),
        "k_levels": int(k),
        "residual_shapiro_p": float(shapiro_p),
    }

    # choose path
    if (not np.isnan(shapiro_p)) and (shapiro_p >= ALPHA):
        path = "RM-ANOVA (GG-corrected)"
        omni = rm_anova_oneway(X)
        results.update({
            "omnibus_test": path,
            "F_or_chi2": omni["F"],
            "df1": omni["df1"],
            "df2": omni["df2"],
            "p_omnibus_unc": omni["p_uncorrected"],
            "epsilon_GG": omni["epsilon_GG"],
            "p_omnibus_GG": omni["p_GG"],
            "effect_size": omni["eta_p2"],
            "effect_size_name": "partial_eta2",
        })
        omnibus_p = omni["p_GG"]

        # posthoc paired t-tests (3 pairs)
        pairs = [(0,1), (0,2), (1,2)]
        raw_ps, pair_rows = [], []
        for i, j in pairs:
            a = X[:, i]; b = X[:, j]
            t = stats.ttest_rel(b, a)
            p = float(t.pvalue)
            raw_ps.append(p)
            dz = paired_effect_dz(a, b)
            d = (b - a)
            ci_l, ci_h = mean_ci_t(d)
            pair_rows.append({
                "outcome": outcome_name,
                "pair": f"{loss_order[i]} vs {loss_order[j]}",
                "test": "paired t-test",
                "statistic": float(t.statistic),
                "p_raw": p,
                "mean_diff": float(d.mean()),
                "ci95_low": ci_l,
                "ci95_high": ci_h,
                "cohen_dz": dz,
            })

        p_holm = holm_adjust(np.array(raw_ps))
        for r, pc in zip(pair_rows, p_holm):
            r["p_holm"] = float(pc)

    else:
        path = "Friedman"
        omni = friedman_test(X)
        results.update({
            "omnibus_test": path,
            "F_or_chi2": omni["chi2"],
            "df1": omni["df"],
            "df2": np.nan,
            "p_omnibus_unc": omni["p"],
            "epsilon_GG": np.nan,
            "p_omnibus_GG": omni["p"],  # same
            "effect_size": omni["kendalls_W"],
            "effect_size_name": "kendalls_W",
        })
        omnibus_p = omni["p"]

        # posthoc Wilcoxon signed-rank (3 pairs)
        pairs = [(0,1), (0,2), (1,2)]
        raw_ps, pair_rows = [], []
        for i, j in pairs:
            a = X[:, i]; b = X[:, j]
            d = b - a
            w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            p = float(w.pvalue)
            raw_ps.append(p)
            dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
            ci_l, ci_h = mean_ci_t(d)  # mean CI (kept consistent with previous reporting)
            pair_rows.append({
                "outcome": outcome_name,
                "pair": f"{loss_order[i]} vs {loss_order[j]}",
                "test": "Wilcoxon signed-rank",
                "statistic": float(w.statistic),
                "p_raw": p,
                "mean_diff": float(d.mean()),
                "ci95_low": ci_l,
                "ci95_high": ci_h,
                "cohen_dz": dz,
            })

        p_holm = holm_adjust(np.array(raw_ps))
        for r, pc in zip(pair_rows, p_holm):
            r["p_holm"] = float(pc)

    return results, pair_rows

omni_rows = []
posthoc_rows = []

res_acc, post_acc = analyze_outcome(wide_acc, "Tuned Accuracy")
res_f1,  post_f1  = analyze_outcome(wide_f1,  "Tuned Macro-F1")

omni_rows += [res_acc, res_f1]
posthoc_rows += post_acc + post_f1

omni_df = pd.DataFrame(omni_rows)
post_df = pd.DataFrame(posthoc_rows)

omni_df.to_csv(OUT_DIR / "omnibus_loss.csv", index=False)
post_df.to_csv(OUT_DIR / "posthoc_loss_holm.csv", index=False)

# =========================
# Figures (2 plots)
# =========================
# prepare significant pairs list per outcome for annotation
def sig_pairs_from_posthoc(post_df, outcome):
    # returns list of (i, j, p_holm)
    pairs = []
    for _, r in post_df[post_df["outcome"] == outcome].iterrows():
        a, b = r["pair"].split(" vs ")
        i = loss_order.index(a)
        j = loss_order.index(b)
        pairs.append((i, j, float(r["p_holm"])))
    # sort by p for consistent stacking
    pairs.sort(key=lambda x: x[2])
    return pairs

sig_acc = sig_pairs_from_posthoc(post_df, "Tuned Accuracy")
sig_f1  = sig_pairs_from_posthoc(post_df, "Tuned Macro-F1")

acc_data = [wide_acc[c].to_numpy() for c in loss_order]
f1_data  = [wide_f1[c].to_numpy()  for c in loss_order]

fig = plt.figure(figsize=(10, 5))
ax = plt.gca()
boxplot_with_sig(
    ax,
    acc_data,
    loss_order,
    title="Tuned Accuracy across loss functions (subject-level)",
    ylabel="Tuned Accuracy",
    sig_pairs=sig_acc
)
plt.tight_layout()
fig.savefig(OUT_DIR / "box_tuned_accuracy_loss.png", dpi=300)
fig.savefig(OUT_DIR / "box_tuned_accuracy_loss.pdf")
plt.close(fig)

fig = plt.figure(figsize=(10, 5))
ax = plt.gca()
boxplot_with_sig(
    ax,
    f1_data,
    loss_order,
    title="Tuned Macro-F1 across loss functions (subject-level)",
    ylabel="Tuned Macro-F1",
    sig_pairs=sig_f1
)
plt.tight_layout()
fig.savefig(OUT_DIR / "box_tuned_macrof1_loss.png", dpi=300)
fig.savefig(OUT_DIR / "box_tuned_macrof1_loss.pdf")
plt.close(fig)

print("Done. Saved to:", OUT_DIR)
print("Omnibus:", OUT_DIR / "omnibus_loss.csv")
print("Posthoc:", OUT_DIR / "posthoc_loss_holm.csv")
