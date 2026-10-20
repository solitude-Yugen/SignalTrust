"""Step 3: cross-dataset test of the SignalTrust gate on BIDMC PPG & Respiration (train on WESAD, test on BIDMC).

Differences from the WESAD pipeline (all deliberate, all stated in the README):
  * PPG-only model: no accelerometer in BIDMC, and `bvp_std` is dropped because PLETH (normalised 0-1) and the
    Empatica BVP use different units. The 5 scale-free features are: snr, skew, kurt, ibi_cv, tmpl_corr.
  * PLETH is resampled 125 -> 64 Hz and the SAME phase-1 code is reused (band-pass, 8 s / 2 s windows, peaks, HR).
  * Reference HR: ECG lead II, upsampled to 700 Hz and passed through the SAME R-peak detector used for WESAD.
  * usable = |HR_ppg - HR_ecg| <= 5 bpm (same rule). Threshold stays at the Step-1 value (0.70), not re-tuned.
BIDMC has little real motion, so the second half injects the same synthetic PPG-only artifacts as stress_test.py.

Usage (Kaggle or local):
  python src/bidmc_eval.py --root /kaggle/input/datasets/jain2026/bidmc --windows out/windows.csv \
         --oof out/oof_predictions.csv --out out
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1 as p1
import stress_test as st
from phase2 import hgb, ece, mae_curve, FR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

F5 = ["snr", "skew", "kurt", "ibi_cv", "tmpl_corr"]
ART = ["motion_ppgonly", "dropout", "saturation", "wander"]       # no coupled-motion: BIDMC has no accelerometer
GATES = ["model", "snr"]
COV_IDX = [2, 4, 6, 8, 9]                                          # 30 / 50 / 70 / 90 / 100 % coverage


# ----------------------------------------------------------------------------- BIDMC loading
def load_record(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]                     # BIDMC headers have leading spaces (' PLETH', ' II')
    d = d.interpolate(limit_direction="both")
    fs = 1 / np.median(np.diff(d["Time [s]"].values))
    assert abs(fs - 125) < 1, f"unexpected sampling rate {fs} in {path}"
    rid = "B" + re.search(r"bidmc_(\d+)_Signals", os.path.basename(path)).group(1)
    return dict(subject=rid,
                raw64=resample_poly(d["PLETH"].values.astype(float), 64, 125),       # 125 -> 64 Hz
                ecg700=resample_poly(d["II"].values.astype(float), 28, 5))           # 125 -> 700 Hz (reuse ECG detector)


def process_record(rec):
    pk_ecg = p1.ecg_peaks(rec["ecg700"])
    x = p1.bandpass(rec["raw64"], 0.5, 8, p1.FS_BVP, 3)
    T = int(min(len(x) / p1.FS_BVP, len(rec["ecg700"]) / p1.FS_ECG))
    aw = np.zeros((p1.WIN_S * p1.FS_ACC, 3))                       # dummy ACC: the ACC features are not used
    rows = []
    for t0 in range(0, T - p1.WIN_S + 1, p1.STEP_S):
        xw = x[t0 * p1.FS_BVP:(t0 + p1.WIN_S) * p1.FS_BVP]
        pk = p1.ppg_peaks(xw)
        hr = p1.hr_from_ibis(np.diff(pk) / p1.FS_BVP)
        r = dict(subject=rec["subject"], t0=t0, hr_ecg=p1.hr_ecg_window(pk_ecg, t0, t0 + p1.WIN_S), hr_ppg=hr)
        r.update(p1.quality_features(xw, pk, hr, aw))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["err"] = (df.hr_ppg - df.hr_ecg).abs()
    df["usable"] = np.where(df.err.notna(), (df.err <= p1.TOL_BPM).astype(float), np.nan)
    return df, T


# ----------------------------------------------------------------------------- model
def fit5(train):
    X, y, g = train[F5].values, train.usable.values.astype(int), train.subject.values
    inner = list(GroupKFold(5).split(X, y, g))
    return CalibratedClassifierCV(hgb(), method="isotonic", cv=inner).fit(X, y)


def wesad_loso_reference(ev):
    """In-domain reference: the same 5-feature PPG-only model, leave-one-subject-out on WESAD."""
    p = np.zeros(len(ev))
    for tr, te in LeaveOneGroupOut().split(ev, groups=ev.subject.values):
        p[te] = fit5(ev.iloc[tr]).predict_proba(ev.iloc[te][F5].values)[:, 1]
        print("  LOSO held out", ev.subject.values[te][0], flush=True)
    y = ev.usable.values.astype(int)
    c = mae_curve(ev.err.values, p)
    return dict(AUROC=roc_auc_score(y, p), MAE50=c[4], MAE30=c[2], MAE70=c[6], ungated=ev.err.mean(), ECE=ece(y, p))


# ----------------------------------------------------------------------------- clean evaluation
def clean_eval(B, p, thr, out):
    y = B.usable.values.astype(int)
    rng = np.random.default_rng(0)
    scores = {"random": rng.random(len(B)), "SNR": B.snr.values, "template corr": B.tmpl_corr.values,
              "IBI regularity": -B.ibi_cv.fillna(1.0).values, "model (WESAD-trained, PPG-only)": p}
    print(f"\nBIDMC windows with estimate+reference: {len(B)} | ungated MAE {B.err.mean():.2f} bpm | "
          f"usable rate {y.mean():.3f} | records {B.subject.nunique()}")
    print("\n=== Quality classification on BIDMC (usable vs not) ===")
    rows = []
    for k, s in scores.items():
        rows.append(dict(gate=k, AUROC=roc_auc_score(y, s) if len(set(y)) > 1 else np.nan))
    M = pd.DataFrame(rows)
    M.loc[M.gate.str.startswith("model"), "ECE"] = ece(y, p)
    print(M.round(3).to_string(index=False)); M.to_csv(os.path.join(out, "bidmc_metrics.csv"), index=False)
    curves = {k: mae_curve(B.err.values, s) for k, s in scores.items()}
    C = pd.DataFrame(curves, index=FR).rename_axis("coverage"); C.to_csv(os.path.join(out, "bidmc_mae_vs_coverage.csv"))
    print(f"\n=== Downstream HR MAE (bpm) on BIDMC; ungated = {B.err.mean():.2f} ===")
    print(C.iloc[COV_IDX].round(2).to_string())
    print("\n=== Per-record MAE at 50% coverage (mean +- std over records) ===")
    for k in ("SNR", "model (WESAD-trained, PPG-only)"):
        v = [d.sort_values("s", ascending=False, kind="stable").err.values for _, d in B.assign(s=scores[k]).groupby("subject")]
        v = [e[:max(1, int(round(0.5 * len(e))))].mean() for e in v]
        print(f"{k:<34} {np.mean(v):.2f} +- {np.std(v):.2f}")
    print("ungated per record:", f"{B.groupby('subject').err.mean().mean():.2f} +- {B.groupby('subject').err.mean().std():.2f}")
    print("\n=== Operating points (calibrated WESAD model, threshold NOT re-tuned) ===")
    for t in (0.5, thr):
        keep = p >= t
        fa = (keep & (y == 0)).sum() / len(y)
        print(f"p>={t}: coverage={keep.mean():.3f}  gated MAE={B.err[keep].mean():.2f}  false-accept rate={fa:.3f}  "
              f"precision(usable)={y[keep].mean():.3f}")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 4))
    for k, c in curves.items():
        plt.plot(FR * 100, c, marker="o", ms=3, lw=2.4 if k.startswith("model") else 1.4, label=k)
    plt.xlabel("Coverage kept (%)"); plt.ylabel("HR MAE (bpm)")
    plt.title("BIDMC cross-dataset: trained on WESAD, tested on BIDMC"); plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out, "bidmc_mae_vs_coverage.png"), dpi=150); plt.close()


# ----------------------------------------------------------------------------- synthetic artifacts on BIDMC
def stress(recs, B, model, thr, thr_snr, n_win, out):
    rows, same = [], []
    accw = np.zeros((p1.WIN_S * p1.FS_ACC, 3))
    for rec in recs:
        rid = rec["subject"]; sid = int(rid[1:])
        T = int(min(len(rec["raw64"]) / p1.FS_BVP, len(rec["ecg700"]) / p1.FS_ECG))
        ok = B[(B.subject == rid) & (B.usable == 1) & (B.t0 >= st.PAD_S) & (B.t0 + p1.WIN_S + st.PAD_S <= T)]
        if len(ok) > n_win:
            ok = ok.sample(n_win, random_state=st.SEED)
        for _, w in ok.sort_values("t0").iterrows():
            t0 = int(w.t0)
            raw = rec["raw64"][(t0 - st.PAD_S) * p1.FS_BVP:(t0 + p1.WIN_S + st.PAD_S) * p1.FS_BVP]
            f0, x0 = st.window_features(raw, accw, st.PAD_S, w.hr_ecg)
            same.append(abs(f0["hr_ppg"] - w.hr_ppg) <= 1.0 if not np.isnan(f0["hr_ppg"]) else False)
            rows.append(dict(subject=rid, t0=t0, artifact="clean", severity=0, hr_ecg=w.hr_ecg, **f0))
            for ai, kind in enumerate(ART):
                for lv in (1, 2, 3, 4):
                    rng = np.random.default_rng([st.SEED, sid, t0, ai, lv])
                    r2, _ = st.inject(kind, lv, raw, accw, x0, st.PAD_S, rng)
                    f, _ = st.window_features(r2, accw, st.PAD_S, w.hr_ecg)
                    rows.append(dict(subject=rid, t0=t0, artifact=kind, severity=lv, hr_ecg=w.hr_ecg, **f))
    R = pd.DataFrame(rows)
    print(f"\nSTRESS: {len(R)} window-conditions from {R.subject.nunique()} records | "
          f"clean recompute matches bidmc windows HR (within 1 bpm): {np.mean(same):.3f}")
    R["p_cal"] = model.predict_proba(R[F5].values)[:, 1]
    R["keep_model"] = R.p_cal >= thr
    R["keep_snr"] = R.snr >= thr_snr
    st.GATES = GATES                                               # summarize() loops over st.GATES
    S = R.groupby(["artifact", "severity"]).apply(st.summarize).reset_index()
    S.to_csv(os.path.join(out, "bidmc_stress_summary.csv"), index=False)
    R[["subject", "t0", "artifact", "severity", "hr_ecg", "hr_ppg", "err", "snr", "p_cal", "keep_model", "keep_snr"]].to_csv(
        os.path.join(out, "bidmc_stress_windows.csv"), index=False)
    c = S[S.artifact == "clean"].iloc[0]
    print(f"CLEAN (n={int(c.n)}): ungated MAE {c.ungated_mae:.2f} | model coverage {c.coverage_model:.3f}, gated MAE "
          f"{c.gated_mae_model:.2f} | SNR-gate coverage {c.coverage_snr:.3f}, gated MAE {c.gated_mae_snr:.2f}")
    cols = ["artifact", "severity", "n", "bad_rate", "no_estimate", "ungated_mae", "coverage_model", "recall_bad_model",
            "false_accept_model", "gated_mae_model", "coverage_snr", "recall_bad_snr", "gated_mae_snr"]
    pd.set_option("display.width", 220)
    print(S[S.artifact != "clean"][cols].round(3).to_string(index=False))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(model="tab:red", snr="tab:green")
    lab = dict(model=f"model (p>={thr:g})", snr="SNR gate")
    fig, axs = plt.subplots(2, len(ART), figsize=(13, 6.2))
    for j, a in enumerate(ART):
        d = S[S.artifact == a].sort_values("severity"); xs = [0] + list(d.severity)
        for k in GATES:
            axs[0, j].plot(xs, [1 - c[f"coverage_{k}"]] + list(1 - d[f"coverage_{k}"]), marker="o", color=col[k],
                           lw=2.4 if k == "model" else 1.4, label=lab[k])
            axs[1, j].plot(xs, [c[f"gated_mae_{k}"]] + list(d[f"gated_mae_{k}"]), marker="o", color=col[k],
                           lw=2.4 if k == "model" else 1.4, label=lab[k])
        axs[0, j].plot(xs, [c.bad_rate] + list(d.bad_rate), "k--", lw=1, label="truly degraded")
        axs[1, j].plot(xs, [c.ungated_mae] + list(d.ungated_mae), "k--", marker="o", label="ungated")
        axs[0, j].set_title(st.DESC[a], fontsize=9); axs[0, j].set_ylim(-0.02, 1.02)
        axs[0, j].set_ylabel("share with NO HR output"); axs[1, j].set_ylabel("HR MAE on output windows (bpm)")
        for i in (0, 1):
            axs[i, j].set_xlabel("severity (0 = clean)"); axs[i, j].set_xticks(range(5)); axs[i, j].grid(alpha=.3)
    axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
    fig.suptitle("BIDMC + synthetic PPG artifacts (WESAD-trained model)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "bidmc_stress.png"), dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing bidmc_csv/bidmc_XX_Signals.csv (searched recursively)")
    ap.add_argument("--windows", default="out/windows.csv", help="WESAD windows.csv from phase 1 (training data)")
    ap.add_argument("--oof", default="out/oof_predictions.csv")
    ap.add_argument("--out", default="out")
    ap.add_argument("--thr", type=float, default=0.70)
    ap.add_argument("--n-windows", type=int, default=60, help="clean windows per record for the artifact test")
    ap.add_argument("--records", type=int, default=None, help="use only the first N records (quick trial)")
    ap.add_argument("--skip-loso", action="store_true", help="skip the WESAD in-domain 5-feature LOSO reference (slow)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    W = pd.read_csv(a.windows); ev = W[W.err.notna()].reset_index(drop=True)
    cov_target = float((pd.read_csv(a.oof).p_cal >= a.thr).mean()) if os.path.exists(a.oof) else 0.47
    thr_snr = float(np.quantile(ev.snr, 1 - cov_target))           # SNR gate matched to the model's WESAD coverage

    paths = sorted(glob.glob(os.path.join(a.root, "**", "bidmc_*_Signals.csv"), recursive=True))
    assert paths, f"no bidmc_*_Signals.csv under {a.root}"
    paths = paths[:a.records] if a.records else paths
    recs, dfs = [], []
    for p in paths:
        r = load_record(p); d, T = process_record(r); recs.append(r); dfs.append(d)
        print(f"{r['subject']}: {len(d)} windows, usable={np.nanmean(d.usable):.2f}, no-estimate={d.err.isna().mean():.2f}", flush=True)
    A = pd.concat(dfs, ignore_index=True); A.to_csv(os.path.join(a.out, "bidmc_windows.csv"), index=False)
    B = A[A.err.notna()].reset_index(drop=True)

    if not a.skip_loso:
        print("\nWESAD in-domain reference (5 PPG-only features, leave-one-subject-out):")
        ref = wesad_loso_reference(ev)
        print({k: round(float(v), 3) for k, v in ref.items()})
        pd.DataFrame([ref]).to_csv(os.path.join(a.out, "wesad_5feat_loso.csv"), index=False)
    model = fit5(ev)                                               # final model: all 15 WESAD subjects
    p = model.predict_proba(B[F5].values)[:, 1]
    B["p_cal"] = p
    B.to_csv(os.path.join(a.out, "bidmc_scored_windows.csv"), index=False)
    clean_eval(B, p, a.thr, a.out)
    stress(recs, B, model, a.thr, thr_snr, a.n_windows, a.out)
    print(f"\nwrote bidmc_*.csv / bidmc_*.png to {a.out}")


if __name__ == "__main__":
    main()
