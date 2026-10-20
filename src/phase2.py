"""Phase 2: leave-one-subject-out quality model + calibrated gate, compared with baseline gates.

Usage: python src/phase2.py --windows out/windows.csv --out out
Inputs to the model: BVP/ACC quality features only (FEATS). hr_ecg / err / usable / label are never features.
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

FEATS = ["snr", "skew", "kurt", "ibi_cv", "tmpl_corr", "acc_std", "acc_range", "bvp_std"]
FR = np.linspace(0.1, 1.0, 10)


def hgb():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=24,
                                          l2_regularization=1.0, random_state=0)


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def mae_curve(err, score):
    e = np.asarray(err)[np.argsort(-np.asarray(score), kind="stable")]
    return np.array([e[:max(1, int(round(f * len(e))))].mean() for f in FR])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="out/windows.csv")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.windows)
    ev = df[df.err.notna()].reset_index(drop=True)
    X, y, g = ev[FEATS].values, ev.usable.values.astype(int), ev.subject.values
    p_raw, p_cal = np.zeros(len(ev)), np.zeros(len(ev))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = hgb().fit(X[tr], y[tr]); p_raw[te] = m.predict_proba(X[te])[:, 1]
        inner = list(GroupKFold(5).split(X[tr], y[tr], g[tr]))
        c = CalibratedClassifierCV(hgb(), method="isotonic", cv=inner).fit(X[tr], y[tr])
        p_cal[te] = c.predict_proba(X[te])[:, 1]
        print("held out", g[te][0], flush=True)
    ev["p_raw"], ev["p_cal"] = p_raw, p_cal
    ev[["subject", "t0", "hr_ecg", "hr_ppg", "err", "usable", "label", "acc_std", "snr", "p_raw", "p_cal"]].to_csv(
        os.path.join(a.out, "oof_predictions.csv"), index=False)

    # classification metrics (pooled + per-subject mean/std)
    print("\n=== Quality classification (leave-one-subject-out) ===")
    rows = []
    for name, p in [("model (raw)", p_raw), ("model (calibrated)", p_cal)]:
        pred = (p >= 0.5).astype(int)
        per = ev.assign(p=p, pred=pred).groupby("subject").apply(
            lambda d: pd.Series(dict(auroc=roc_auc_score(d.usable, d.p), f1m=f1_score(d.usable, d.pred, average="macro"))))
        rows.append(dict(model=name, AUROC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                         macroF1=f1_score(y, pred, average="macro"), F1_unusable=f1_score(1 - y, 1 - pred),
                         Brier=brier_score_loss(y, p), ECE=ece(y, p),
                         subj_AUROC=f"{per.auroc.mean():.3f}+-{per.auroc.std():.3f}",
                         subj_macroF1=f"{per.f1m.mean():.3f}+-{per.f1m.std():.3f}"))
    for k, base in [("baseline: SNR", ev.snr.values), ("baseline: accel", -ev.acc_std.values)]:
        rows.append(dict(model=k, AUROC=roc_auc_score(y, base), PR_AUC=average_precision_score(y, base)))
    M = pd.DataFrame(rows); M.to_csv(os.path.join(a.out, "model_metrics.csv"), index=False)
    print(M.round(3).to_string(index=False))

    # downstream: MAE vs coverage
    rng = np.random.default_rng(0)
    scores = {"random": rng.random(len(ev)), "accel": -ev.acc_std.values, "SNR": ev.snr.values,
              "model (calibrated)": p_cal}
    curves = {k: mae_curve(ev.err.values, s) for k, s in scores.items()}
    C = pd.DataFrame(curves, index=FR).rename_axis("coverage"); C.to_csv(os.path.join(a.out, "model_mae_vs_coverage.csv"))
    print(f"\n=== Downstream HR MAE (bpm); ungated = {ev.err.mean():.2f} ===")
    print(C.iloc[[2, 4, 6, 8, 9]].round(2).to_string())

    # per-subject MAE at 50% / 70% coverage (ranking within each subject)
    print("\n=== Per-subject MAE at coverage (mean +- std over 15 subjects) ===")
    for cov in (0.5, 0.7):
        out = {}
        for k, s in scores.items():
            vals = []
            for sub, d in ev.assign(s=s).groupby("subject"):
                e = d.sort_values("s", ascending=False).err.values
                vals.append(e[:max(1, int(round(cov * len(e))))].mean())
            out[k] = f"{np.mean(vals):.2f}+-{np.std(vals):.2f}"
        print(f"coverage {int(cov*100)}%:", out)
    print("ungated per-subject:", f"{ev.groupby('subject').err.mean().mean():.2f}+-{ev.groupby('subject').err.mean().std():.2f}")

    # operating points + false accepts
    print("\n=== Operating points (calibrated model) ===")
    for t in (0.5, 0.6, 0.7, 0.8):
        keep = p_cal >= t
        fa = ((keep) & (y == 0)).sum() / len(y)
        print(f"p>={t}: coverage={keep.mean():.2f}  gated MAE={ev.err[keep].mean():.2f}  "
              f"false-accept rate={fa:.3f}  precision(usable)={y[keep].mean():.3f}")

    # protocol-only view (WESAD labels 1-4) as a robustness check
    pr = ev.label.isin([1, 2, 3, 4]).values
    print(f"\n=== Protocol windows only (labels 1-4, n={pr.sum()}) ===")
    for k, s in scores.items():
        c = mae_curve(ev.err.values[pr], s[pr]); print(f"{k:<20} MAE@50%={c[4]:.2f}  MAE@100%={c[-1]:.2f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 4))
    for k, c in curves.items():
        plt.plot(FR * 100, c, marker="o", ms=3, lw=2.4 if k.startswith("model") else 1.4, label=k)
    plt.xlabel("Coverage kept (%)"); plt.ylabel("HR MAE (bpm)")
    plt.title("Quality-gated HR error (leave-one-subject-out)"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.out, "model_mae_vs_coverage.png"), dpi=150)


if __name__ == "__main__":
    main()
