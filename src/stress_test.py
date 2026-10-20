"""Step 2: robustness stress test for the SignalTrust quality gate.

For each held-out subject: take clean, quiet-wrist, usable windows from the raw recording, inject synthetic
artifacts at 4 severity levels, recompute the 8 quality features with the exact phase-1 code, score them with the
leave-one-subject-out model (trained WITHOUT that subject), and report how the gate and the gated HR error behave.

Usage (from the project root):
  python src/stress_test.py --root <dir with S2..S17> --windows out/windows.csv --oof out/oof_predictions.csv --out out

A window yields an HR only if the gate passes AND the PPG estimator returns a value; otherwise the system abstains
(this is what "coverage", "rejected" and "abstain" mean below). Everything injected here is SYNTHETIC. This measures how the gate reacts to controlled artifacts applied to clean
windows; it is a sensitivity study, not a population estimate of real-world performance.
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1 as p1
from phase2 import hgb, FEATS
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

PAD_S = 6                    # context (s) on each side so the band-pass has no edge effect inside the window
QUIET_ACC = 0.3              # same "quiet" definition as the estimator study (acc_std < 0.3)
SEED = 0
ARTIFACTS = ["motion", "motion_ppgonly", "dropout", "saturation", "wander"]
DESC = {
    "motion": "Motion burst (PPG + accelerometer coupled)",
    "motion_ppgonly": "Motion burst (PPG only, ACC untouched)",
    "dropout": "Sensor dropout (flat segment)",
    "saturation": "Saturation / clipping",
    "wander": "Baseline wander (0.3-0.45 Hz)",
}
# severity 1..4, increasing
LEVELS = {
    "motion": dict(ppg=[0.5, 1, 2, 4], acc=[1, 3, 6, 12]),       # PPG burst amp = x * window std; target acc_std (raw units)
    "motion_ppgonly": dict(ppg=[0.5, 1, 2, 4]),
    "dropout": dict(frac=[0.1, 0.2, 0.4, 0.6]),                  # fraction of the 8 s window held flat
    "saturation": dict(c=[1.0, 0.6, 0.35, 0.15]),                # clip at median +- c * half (p5..p95) range
    "wander": dict(k=[2, 5, 15, 50]),                            # sinusoid amp = k * window std (before band-pass)
}
GATES = ["model", "snr", "accel"]


# ----------------------------------------------------------------------------- artifacts
def burst_params(rng):
    return dict(start=rng.uniform(0, 0.5), f=rng.uniform(1.2, 3.0),
                ph1=rng.uniform(0, 2 * np.pi), ph2=rng.uniform(0, 2 * np.pi), seed=int(rng.integers(1 << 31)))


def burst(bp, n, fs, frac=0.5):
    """Unit-std motion burst of n samples at rate fs, active over a contiguous `frac` of the window."""
    L, s = int(frac * n), int(bp["start"] * n)
    t = np.arange(L) / fs
    r = np.random.default_rng(bp["seed"])
    b = (np.sin(2 * np.pi * bp["f"] * t + bp["ph1"]) + 0.5 * np.sin(4 * np.pi * bp["f"] * t + bp["ph2"])
         + 0.3 * r.standard_normal(L))
    b = b / b.std()
    from scipy.signal.windows import tukey
    out = np.zeros(n)
    out[s:s + L] = b * tukey(L, 0.3)
    return out


def add_acc_burst(aw, b, target, rng):
    """Add a burst along a mostly-gravity direction, scaled so the window's |acc| std is ~target (raw units)."""
    g = aw.mean(0); g = g / (np.linalg.norm(g) + 1e-9)
    o = rng.standard_normal(3); o -= o.dot(g) * g; o /= (np.linalg.norm(o) + 1e-9)
    u = 0.8 * g + 0.6 * o
    a = float(target)
    for _ in range(6):
        m = np.linalg.norm(aw + a * b[:, None] * u, axis=1).std()
        if m < 1e-9:
            break
        a = min(a * target / m, 500.0)
    return aw + a * b[:, None] * u


def inject(kind, level, raw, acc, x0_std, pad, rng):
    """raw: padded raw BVP segment (pad + 8 s + pad); acc: (8 s * 32 Hz, 3). Returns modified copies."""
    raw, acc = raw.copy(), acc.copy()
    i0, i1 = pad * p1.FS_BVP, (pad + p1.WIN_S) * p1.FS_BVP
    n = i1 - i0
    if kind in ("motion", "motion_ppgonly"):
        bp = burst_params(rng)
        raw[i0:i1] += LEVELS[kind]["ppg"][level - 1] * x0_std * burst(bp, n, p1.FS_BVP)
        if kind == "motion":
            acc = add_acc_burst(acc, burst(bp, len(acc), p1.FS_ACC), LEVELS[kind]["acc"][level - 1], rng)
    elif kind == "dropout":
        L = int(LEVELS[kind]["frac"][level - 1] * n)
        s = i0 + int(rng.integers(0, n - L + 1))
        raw[s:s + L] = raw[s - 1]                       # sensor holds its last value
    elif kind == "saturation":
        seg = raw[i0:i1]
        m, h = np.median(seg), (np.percentile(seg, 95) - np.percentile(seg, 5)) / 2
        c = LEVELS[kind]["c"][level - 1]
        raw[i0:i1] = np.clip(seg, m - c * h, m + c * h)
    elif kind == "wander":
        t = np.arange(len(raw)) / p1.FS_BVP
        raw += LEVELS[kind]["k"][level - 1] * x0_std * np.sin(2 * np.pi * rng.uniform(0.3, 0.45) * t + rng.uniform(0, 6.28))
    return raw, acc


# ----------------------------------------------------------------------------- features (exact phase-1 code)
def window_features(raw_seg, acc_win, pad, hr_ecg):
    x = p1.bandpass(raw_seg, 0.5, 8, p1.FS_BVP, 3)[pad * p1.FS_BVP:(pad + p1.WIN_S) * p1.FS_BVP]
    pk = p1.ppg_peaks(x)
    hr = p1.hr_from_ibis(np.diff(pk) / p1.FS_BVP)
    f = p1.quality_features(x, pk, hr, acc_win)
    f.update(hr_ppg=hr, err=abs(hr - hr_ecg) if not np.isnan(hr) else np.nan)
    return f, float(x.std())


def fit_gate(train):
    """Same model recipe as phase2: HGB + isotonic calibration with inner GroupKFold(5)."""
    X, y, g = train[FEATS].values, train.usable.values.astype(int), train.subject.values
    inner = list(GroupKFold(5).split(X, y, g))
    return CalibratedClassifierCV(hgb(), method="isotonic", cv=inner).fit(X, y)


# ----------------------------------------------------------------------------- summaries
def summarize(g):
    est = g.err.notna()
    bad = (g.err > p1.TOL_BPM) | ~est                       # degraded window: wrong HR or no HR at all
    out = dict(n=len(g), no_estimate=(~est).mean(), bad_rate=bad.mean(), ungated_mae=g.err.mean())
    for k in GATES:
        keep = g[f"keep_{k}"].astype(bool) & est             # HR is output only if the gate passes AND an estimate exists
        out[f"coverage_{k}"] = keep.mean()
        out[f"recall_bad_{k}"] = (~keep)[bad].mean() if bad.any() else np.nan
        out[f"false_accept_{k}"] = (keep & (g.err > p1.TOL_BPM)).mean()
        out[f"gated_mae_{k}"] = g.err[keep].mean() if keep.any() else np.nan
    return pd.Series(out)


def make_plots(S, out, thr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clean = S[S.artifact == "clean"].iloc[0]
    col = dict(model="tab:red", snr="tab:green", accel="tab:orange")
    lab = dict(model=f"model (p>={thr:g})", snr="SNR gate", accel="accel gate")
    for fname, kind in (("stress_rejection.png", "rej"), ("stress_mae.png", "mae")):
        fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharey=(kind == "rej"))
        for ax, a in zip(axs.ravel(), ARTIFACTS):
            d = S[S.artifact == a].sort_values("severity")
            xs = [0] + list(d.severity)
            if kind == "rej":
                for k in GATES:
                    ax.plot(xs, [1 - clean[f"coverage_{k}"]] + list(1 - d[f"coverage_{k}"]), marker="o",
                            color=col[k], lw=2.4 if k == "model" else 1.4, label=lab[k])
                ax.plot(xs, [clean.bad_rate] + list(d.bad_rate), "k--", lw=1, label="truly degraded (err>5 or no HR)")
                ax.set_ylabel("share of windows with NO HR output"); ax.set_ylim(-0.02, 1.02)
            else:
                ax.plot(xs, [clean.ungated_mae] + list(d.ungated_mae), "k--", marker="o", label="ungated")
                for k in GATES:
                    ax.plot(xs, [clean[f"gated_mae_{k}"]] + list(d[f"gated_mae_{k}"]), marker="o", color=col[k],
                            lw=2.4 if k == "model" else 1.4, label=lab[k])
                ax.set_ylabel("HR MAE on accepted windows (bpm)")
            ax.set_title(DESC[a], fontsize=9); ax.set_xlabel("severity (0 = clean)"); ax.set_xticks(range(5)); ax.grid(alpha=.3)
        h, l = axs[0, 0].get_legend_handles_labels()
        axs[1, 2].axis("off"); axs[1, 2].legend(h, l, loc="center", fontsize=9)
        fig.suptitle("Stress test: abstention vs artifact severity (synthetic artifacts on clean quiet windows)" if kind == "rej"
                     else "Stress test: HR error on windows each gate accepts (synthetic artifacts)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(out, fname), dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing S*/S*.pkl")
    ap.add_argument("--windows", default="out/windows.csv")
    ap.add_argument("--oof", default="out/oof_predictions.csv", help="used only to match baseline-gate coverage")
    ap.add_argument("--out", default="out")
    ap.add_argument("--thr", type=float, default=0.70, help="abstain threshold on calibrated P(usable) (Step 1)")
    ap.add_argument("--n-windows", type=int, default=100, help="clean quiet windows per subject")
    ap.add_argument("--subjects", nargs="*", default=None, help="e.g. S2 S3 (default: all found)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.windows)
    ev = df[df.err.notna()].reset_index(drop=True)
    cov_target = float((pd.read_csv(a.oof).p_cal >= a.thr).mean()) if os.path.exists(a.oof) else 0.47
    print(f"model coverage at p>={a.thr}: {cov_target:.3f}  (baseline gates are matched to this coverage)")

    paths = sorted(glob.glob(os.path.join(a.root, "S*", "S*.pkl")))
    assert paths, f"no S*/S*.pkl under {a.root}"
    all_rows = []
    for path in paths:
        sub = p1.load_subject(path)
        name = sub["subject"]
        if a.subjects and name not in a.subjects:
            continue
        sid = int(re.sub(r"\D", "", name) or 0)
        rows = ev[ev.subject == name]
        if rows.empty:                                   # subject not in windows.csv (e.g. synthetic test subject)
            rows = p1.process_subject(sub); rows = rows[rows.err.notna()]
        train = ev[ev.subject != name]
        model = fit_gate(train)
        thr_acc = float(np.quantile(train.acc_std, cov_target))        # accept if acc_std <= thr_acc
        thr_snr = float(np.quantile(train.snr, 1 - cov_target))        # accept if snr >= thr_snr

        T = int(min(len(sub["bvp"]) / p1.FS_BVP, len(sub["acc"]) / p1.FS_ACC, len(sub["ecg"]) / p1.FS_ECG))
        ok = rows[(rows.usable == 1) & (rows.acc_std < QUIET_ACC) & (rows.t0 >= PAD_S) & (rows.t0 + p1.WIN_S + PAD_S <= T)]
        if len(ok) > a.n_windows:
            ok = ok.sample(a.n_windows, random_state=SEED)
        ok = ok.sort_values("t0")
        print(f"{name}: {len(ok)} clean quiet windows | held-out model trained on {train.subject.nunique()} subjects", flush=True)

        recs, same = [], []
        for _, w in ok.iterrows():
            t0 = int(w.t0)
            raw = sub["bvp"][(t0 - PAD_S) * p1.FS_BVP:(t0 + p1.WIN_S + PAD_S) * p1.FS_BVP]
            acc = sub["acc"][t0 * p1.FS_ACC:(t0 + p1.WIN_S) * p1.FS_ACC]
            f0, x0_std = window_features(raw, acc, PAD_S, w.hr_ecg)
            same.append(abs(f0["hr_ppg"] - w.hr_ppg) <= 1.0 if not np.isnan(f0["hr_ppg"]) else False)
            recs.append(dict(subject=name, t0=t0, artifact="clean", severity=0, hr_ecg=w.hr_ecg, **f0))
            for ai, kind in enumerate(ARTIFACTS):
                for lv in (1, 2, 3, 4):
                    rng = np.random.default_rng([SEED, sid, t0, ai, lv])
                    r2, a2 = inject(kind, lv, raw, acc, x0_std, PAD_S, rng)
                    f, _ = window_features(r2, a2, PAD_S, w.hr_ecg)
                    recs.append(dict(subject=name, t0=t0, artifact=kind, severity=lv, hr_ecg=w.hr_ecg, **f))
        if not recs:
            print(f"  {name}: no eligible windows, skipped"); continue
        R = pd.DataFrame(recs)
        print(f"  clean recompute matches windows.csv HR (within 1 bpm): {np.mean(same):.3f}")
        R["p_cal"] = model.predict_proba(R[FEATS].values)[:, 1]
        R["keep_model"] = R.p_cal >= a.thr
        R["keep_snr"] = R.snr >= thr_snr
        R["keep_accel"] = R.acc_std <= thr_acc
        all_rows.append(R)

    W = pd.concat(all_rows, ignore_index=True)
    W[["subject", "t0", "artifact", "severity", "hr_ecg", "hr_ppg", "err", "snr", "acc_std", "p_cal",
       "keep_model", "keep_snr", "keep_accel"]].to_csv(os.path.join(a.out, "stress_windows.csv"), index=False)
    key = ["artifact", "severity"]
    S = W.groupby(key).apply(summarize).reset_index()
    S.to_csv(os.path.join(a.out, "stress_summary.csv"), index=False)
    W.groupby(["subject"] + key).apply(summarize).reset_index().to_csv(os.path.join(a.out, "stress_by_subject.csv"), index=False)

    pd.set_option("display.width", 220)
    c = S[S.artifact == "clean"].iloc[0]
    print(f"\nCLEAN (n={int(c.n)}): ungated MAE {c.ungated_mae:.2f} | model coverage {c.coverage_model:.2f}, "
          f"gated MAE {c.gated_mae_model:.2f}")
    cols = ["artifact", "severity", "n", "bad_rate", "no_estimate", "ungated_mae", "coverage_model", "recall_bad_model",
            "false_accept_model", "gated_mae_model", "recall_bad_snr", "recall_bad_accel", "gated_mae_snr", "gated_mae_accel"]
    print(S[S.artifact != "clean"][cols].round(3).to_string(index=False))
    make_plots(S, a.out, a.thr)
    print(f"\nwrote stress_windows.csv, stress_summary.csv, stress_by_subject.csv, stress_rejection.png, stress_mae.png to {a.out}")


if __name__ == "__main__":
    main()
