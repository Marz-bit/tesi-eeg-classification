# from pathlib import Path
# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt

# # =========================
# # USER INPUTS
# # =========================
# CSV_FLAT = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9 classes\9_FLAT\20260109-160943.csv")     # <-- cambia
# CSV_BEST = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\statistical analysis\Flat\Fake2\20260127-155132 - Copia_TunedAccMean305947.csv")        # <-- cambia

# OUT_DIR = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\statistical analysis\Flat\Fake2")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# SUBJECT_COL = "Subject"

# # Columns expected in EACH CSV (same schema as your other outputs)
# OUTCOMES = [
#     ("Unseen Acc.", "Unseen Accuracy"),
#     ("Unseen F1",   "Unseen Macro-F1"),
#     ("Tuned Acc.",  "Tuned Accuracy"),
#     ("Tuned F1",    "Tuned Macro-F1"),
# ]

# ALPHA = 0.05


# # =========================
# # Helpers
# # =========================
# def detect_sep(p: Path) -> str:
#     head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
#     if "\t" in head: return "\t"
#     if ";" in head:  return ";"
#     return ","


# def to_float(s: pd.Series) -> pd.Series:
#     return pd.to_numeric(
#         s.astype(str).str.strip().str.replace(",", ".", regex=False),
#         errors="coerce"
#     )


# def mean_ci_t(x, conf=0.95):
#     x = np.asarray(x, dtype=float)
#     n = len(x)
#     mu = x.mean()
#     if n < 2:
#         return (np.nan, np.nan)
#     se = x.std(ddof=1) / np.sqrt(n)
#     tcrit = stats.t.ppf((1 + conf) / 2.0, df=n - 1)
#     return mu - tcrit * se, mu + tcrit * se


# def holm_adjust(pvals):
#     """
#     Holm step-down adjustment. Returns adjusted p-values in original order.
#     """
#     pvals = np.asarray(pvals, dtype=float)
#     m = len(pvals)
#     order = np.argsort(pvals)
#     ranked = pvals[order]

#     adj = np.empty(m, dtype=float)
#     # step-down: adj_i = max_{j<=i} ( (m-j+1)*p_(j) )
#     running_max = 0.0
#     for i in range(m):
#         factor = (m - i)
#         val = factor * ranked[i]
#         running_max = max(running_max, val)
#         adj[i] = min(running_max, 1.0)

#     # map back
#     out = np.empty(m, dtype=float)
#     out[order] = adj
#     return out


# def p_to_stars(p):
#     if np.isnan(p): return "n.s."
#     if p < 0.001:   return "***"
#     if p < 0.01:    return "**"
#     if p < 0.05:    return "*"
#     return "n.s."


# def save_both(fig, base: Path):
#     fig.savefig(base.with_suffix(".png"), dpi=300)
#     fig.savefig(base.with_suffix(".pdf"))


# def add_sig_bracket(ax, x1, x2, y, h, text):
#     # bracket from (x1,y) to (x2,y) with height h
#     ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5)
#     ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom")


# def boxplot_two_conditions(values_a, values_b, label_a, label_b, ylab, title, p_corr, test_label, outbase):
#     fig = plt.figure(figsize=(8, 5))
#     ax = plt.gca()

#     bp = ax.boxplot(
#         [values_a, values_b],
#         labels=[label_a, label_b],
#         showmeans=True,
#         meanprops=dict(marker="+", markersize=12),
#         widths=0.5
#     )
#     ax.set_ylabel(ylab)
#     ax.set_title(title)

#     # bracket positioning
#     allv = np.concatenate([values_a, values_b])
#     y_min, y_max = np.nanmin(allv), np.nanmax(allv)
#     yr = (y_max - y_min) if (y_max > y_min) else 1.0
#     y = y_max + 0.06 * yr
#     h = 0.02 * yr

#     stars = p_to_stars(p_corr)
#     if stars != "n.s.":
#         add_sig_bracket(ax, 1, 2, y, h, stars)

#     # clean annotation in the upper-left corner (axes coords)
#     ax.text(
#         0.02, 0.98,
#         f"{test_label}, p(Holm)={p_corr:.3g}",
#         transform=ax.transAxes,
#         ha="left", va="top"
#     )

#     ax.set_ylim(top=y + 0.10 * yr)
#     plt.tight_layout()
#     save_both(fig, outbase)
#     plt.close(fig)


# # =========================
# # Load + subject-level aggregation
# # =========================
# def load_subject_level(csv_path: Path) -> pd.DataFrame:
#     sep = detect_sep(csv_path)
#     df = pd.read_csv(csv_path, sep=sep)

#     # basic cleaning
#     df = df.dropna(subset=[SUBJECT_COL]).copy()

#     # numeric conversion
#     needed_cols = [SUBJECT_COL] + [c for c, _ in OUTCOMES]
#     missing = [c for c in needed_cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

#     for c, _ in OUTCOMES:
#         df[c] = to_float(df[c])

#     df = df.dropna(subset=[c for c, _ in OUTCOMES]).copy()

#     # fold -> subject mean
#     subj = df.groupby(SUBJECT_COL)[[c for c, _ in OUTCOMES]].mean()
#     subj["n_folds"] = df.groupby(SUBJECT_COL).size()
#     return subj


# flat_subj = load_subject_level(CSV_FLAT)
# best_subj = load_subject_level(CSV_BEST)

# # align paired subjects (intersection only)
# common_subjects = flat_subj.index.intersection(best_subj.index)
# flat_subj = flat_subj.loc[common_subjects].sort_index()
# best_subj = best_subj.loc[common_subjects].sort_index()

# if len(common_subjects) < 3:
#     raise RuntimeError("Not enough common subjects to run paired statistics (need >= 3).")

# # =========================
# # Stats per outcome
# # =========================
# rows = []
# raw_pvals = []

# # keep per-outcome objects to plot after Holm correction
# cache = {}

# for col, nice_name in OUTCOMES:
#     x = flat_subj[col].to_numpy(dtype=float)   # baseline (Flat EEGNet)
#     y = best_subj[col].to_numpy(dtype=float)   # Best framework
#     d = y - x
#     n = len(d)

#     shapiro_p = stats.shapiro(d).pvalue if n >= 3 else np.nan

#     if (not np.isnan(shapiro_p)) and (shapiro_p >= ALPHA):
#         test_used = "paired t-test"
#         t = stats.ttest_rel(y, x)
#         stat = float(t.statistic)
#         p = float(t.pvalue)
#     else:
#         test_used = "Wilcoxon signed-rank"
#         w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
#         stat = float(w.statistic)
#         p = float(w.pvalue)

#     dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
#     ci_low, ci_high = mean_ci_t(d, conf=0.95)

#     raw_pvals.append(p)
#     cache[nice_name] = dict(x=x, y=y, d=d, shapiro_p=shapiro_p, test_used=test_used, stat=stat, p_raw=p, dz=dz,
#                             ci_low=ci_low, ci_high=ci_high, n=n)

# # Holm correction across the 4 outcomes
# p_holm = holm_adjust(raw_pvals)

# # Build summary + plots
# paired_values_rows = []
# for i, (col, nice_name) in enumerate(OUTCOMES):
#     info = cache[nice_name]
#     p_corr = float(p_holm[i])

#     rows.append({
#         "outcome": nice_name,
#         "n_subjects": int(info["n"]),
#         "mean_flat": float(np.mean(info["x"])),
#         "std_flat": float(np.std(info["x"], ddof=1)),
#         "mean_best": float(np.mean(info["y"])),
#         "std_best": float(np.std(info["y"], ddof=1)),
#         "mean_delta(best-flat)": float(np.mean(info["d"])),
#         "std_delta": float(np.std(info["d"], ddof=1)),
#         "ci95_delta_low": float(info["ci_low"]),
#         "ci95_delta_high": float(info["ci_high"]),
#         "shapiro_p(delta)": float(info["shapiro_p"]),
#         "test_used": info["test_used"],
#         "statistic": float(info["stat"]),
#         "p_raw": float(info["p_raw"]),
#         "p_holm": p_corr,
#         "cohen_dz": float(info["dz"]),
#     })

#     # paired values long-ish
#     for s, xv, yv, dv in zip(common_subjects, info["x"], info["y"], info["d"]):
#         paired_values_rows.append({
#             "Subject": s,
#             "Outcome": nice_name,
#             "Flat": float(xv),
#             "Best": float(yv),
#             "Delta(Best-Flat)": float(dv),
#         })

#     # figure (two boxplots)
#     out_tag = nice_name.replace(" ", "_").replace("-", "").lower()
#     boxplot_two_conditions(
#         info["x"], info["y"],
#         label_a="Flat EEGNet (9-class)",
#         label_b="Best framework",
#         ylab=nice_name,
#         title=f"{nice_name}: Flat EEGNet vs Best framework (subject-level)",
#         p_corr=p_corr,
#         test_label=info["test_used"],
#         outbase=OUT_DIR / f"box_flat_vs_best_{out_tag}"
#     )

# summary = pd.DataFrame(rows)
# summary.to_csv(OUT_DIR / "summary_subjectlevel_flat_vs_best_stats.csv", index=False)

# paired_df = pd.DataFrame(paired_values_rows)
# paired_df.to_csv(OUT_DIR / "subjectlevel_paired_values_flat_vs_best.csv", index=False)

# print("Done. Results saved in:", OUT_DIR)
# print(summary)


#SOLO TUNED ACCURACY
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

CSV_FLAT = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9 classes\9_FLAT\20260109-160943.csv")
CSV_BEST = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\TREE OF EXPERTS\9_TRAINING\F-1\9_hybrid_iniz_F-1\20260127-155132.csv")

OUT_DIR = Path(r"C:\Users\marzi\Documents\erasmus\Tesi\statistical analysis\Flat\Fake3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_COL = "Subject"
COL_TUNED_ACC = "Tuned Acc."   # cambia se nel CSV hai un nome diverso
ALPHA = 0.05

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


def mean_ci_t(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    if n < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf((1 + conf) / 2.0, df=n - 1)
    return mu - tcrit * se, mu + tcrit * se


def p_to_stars(p):
    if np.isnan(p): return "n.s."
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "n.s."


def save_both(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))


def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5)
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom")


def boxplot_two_conditions(values_a, values_b, label_a, label_b, ylab, title, pval, test_label, outbase):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    ax.boxplot(
        [values_a, values_b],
        labels=[label_a, label_b],
        showmeans=True,
        meanprops=dict(marker="+", markersize=12),
        widths=0.5
    )
    ax.set_ylabel(ylab)
    ax.set_title(title)

    allv = np.concatenate([values_a, values_b])
    y_min, y_max = np.nanmin(allv), np.nanmax(allv)
    yr = (y_max - y_min) if (y_max > y_min) else 1.0
    y = y_max + 0.06 * yr
    h = 0.02 * yr

    st = p_to_stars(pval)
    if st != "n.s.":
        add_sig_bracket(ax, 1, 2, y, h, st)

    ax.text(
        0.02, 0.98,
        f"{test_label}, p={pval:.4g}",
        transform=ax.transAxes,
        ha="left", va="top"
    )

    ax.set_ylim(top=y + 0.10 * yr)
    plt.tight_layout()
    save_both(fig, outbase)
    plt.close(fig)


def load_subject_level(csv_path: Path) -> pd.DataFrame:
    sep = detect_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)

    if SUBJECT_COL not in df.columns:
        raise ValueError(f"Missing '{SUBJECT_COL}' in {csv_path.name}")
    if COL_TUNED_ACC not in df.columns:
        raise ValueError(f"Missing '{COL_TUNED_ACC}' in {csv_path.name}")

    df = df.dropna(subset=[SUBJECT_COL]).copy()
    df[COL_TUNED_ACC] = to_float(df[COL_TUNED_ACC])
    df = df.dropna(subset=[COL_TUNED_ACC]).copy()

    # fold -> subject mean
    subj = df.groupby(SUBJECT_COL)[[COL_TUNED_ACC]].mean()
    subj["n_folds"] = df.groupby(SUBJECT_COL).size()
    return subj


flat_subj = load_subject_level(CSV_FLAT)
best_subj = load_subject_level(CSV_BEST)

# align paired subjects
common_subjects = flat_subj.index.intersection(best_subj.index)
flat_subj = flat_subj.loc[common_subjects].sort_index()
best_subj = best_subj.loc[common_subjects].sort_index()

if len(common_subjects) < 3:
    raise RuntimeError("Not enough common subjects to run paired statistics (need >= 3).")


x = flat_subj[COL_TUNED_ACC].to_numpy(dtype=float)
y = best_subj[COL_TUNED_ACC].to_numpy(dtype=float)
d = y - x
n = len(d)

shapiro_p = stats.shapiro(d).pvalue if n >= 3 else np.nan

if (not np.isnan(shapiro_p)) and (shapiro_p >= ALPHA):
    test_used = "paired t-test"
    t = stats.ttest_rel(y, x)
    stat = float(t.statistic)
    p_raw = float(t.pvalue)
else:
    test_used = "Wilcoxon signed-rank"
    w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    stat = float(w.statistic)
    p_raw = float(w.pvalue)

dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
ci_low, ci_high = mean_ci_t(d, conf=0.95)

summary = pd.DataFrame([{
    "outcome": "Tuned Accuracy",
    "n_subjects": int(n),
    "mean_flat": float(np.mean(x)),
    "std_flat": float(np.std(x, ddof=1)),
    "mean_best": float(np.mean(y)),
    "std_best": float(np.std(y, ddof=1)),
    "mean_delta(best-flat)": float(np.mean(d)),
    "std_delta": float(np.std(d, ddof=1)),
    "ci95_delta_low": float(ci_low),
    "ci95_delta_high": float(ci_high),
    "shapiro_p(delta)": float(shapiro_p),
    "test_used": test_used,
    "statistic": stat,
    "p_raw": p_raw,
    "cohen_dz": float(dz),
}])

summary.to_csv(OUT_DIR / "summary_subjectlevel_flat_vs_best_tunedacc.csv", index=False)

paired = pd.DataFrame({
    "Subject": common_subjects,
    "Flat_TunedAcc": x,
    "Best_TunedAcc": y,
    "Delta(Best-Flat)": d
})
paired.to_csv(OUT_DIR / "subjectlevel_paired_values_flat_vs_best_tunedacc.csv", index=False)

boxplot_two_conditions(
    x, y,
    label_a="Flat EEGNet (9-class)",
    label_b="Best framework",
    ylab="Tuned Accuracy",
    title="Tuned Accuracy: Flat EEGNet vs Best framework (subject-level)",
    pval=p_raw,
    test_label=test_used,
    outbase=OUT_DIR / "box_flat_vs_best_tuned_accuracy"
)

print("Done. Saved to:", OUT_DIR)
print(summary)
