"""Phase 1: WESAD loader, HR extraction, ECG-agreement labels, quality features, baseline gates.

Usage:  python src/phase1.py --root <path containing S2..S17 folders> --out out
Only wrist BVP + wrist ACC are used as model inputs. Chest ECG is used ONLY to build labels.
"""
import argparse, glob, os, pickle
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score

FS_ECG, FS_BVP, FS_ACC = 700, 64, 32
WIN_S, STEP_S, TOL_BPM = 8, 2, 5.0      # window length, stride (s), HR agreement tolerance (bpm)
FEATS = ["snr", "skew", "kurt", "ibi_cv", "tmpl_corr", "acc_std", "acc_range", "bvp_std"]


def load_subject(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    w, c = d["signal"]["wrist"], d["signal"]["chest"]
    return dict(subject=str(d.get("subject", os.path.basename(path)[:-4])),
                ecg=np.asarray(c["ECG"]).ravel().astype(float),
                bvp=np.asarray(w["BVP"]).ravel().astype(float),
                acc=np.asarray(w["ACC"]).astype(float),
                label=np.asarray(d["label"]).ravel())


def bandpass(x, lo, hi, fs, order):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def ecg_peaks(ecg):
    """R-peak indices (700 Hz samples). Simple chunked detector: enough for a reference HR label."""
    x = bandpass(ecg, 5, 15, FS_ECG, 2)
    chunk, out = 30 * FS_ECG, []
    for s in range(0, len(x), chunk):
        seg = x[s:s + chunk]
        if len(seg) < FS_ECG:
            continue
        if skew(seg) < 0:
            seg = -seg
        p, _ = find_peaks(seg, height=0.5 * seg.max(), distance=int(0.3 * FS_ECG))
        out.extend(p + s)
    return np.array(out, dtype=int)


def hr_from_ibis(ibi, min_valid=3):
    ibi = ibi[(ibi > 0.3) & (ibi < 1.5)]
    return 60.0 / np.median(ibi) if len(ibi) >= min_valid else np.nan


def hr_ecg_window(pk, t0, t1):
    sel = pk[(pk >= t0 * FS_ECG) & (pk < t1 * FS_ECG)]
    return hr_from_ibis(np.diff(sel) / FS_ECG)


def ppg_peaks(x):
    # distance 0.4 s (<=150 bpm) + range-based prominence avoids counting the dicrotic notch as a beat
    rng = np.percentile(x, 95) - np.percentile(x, 5)
    p, _ = find_peaks(x, distance=int(0.4 * FS_BVP), prominence=0.4 * rng)
    return p


def quality_features(x, peaks, hr, acc):
    n, f = 2048, np.fft.rfftfreq(2048, 1 / FS_BVP)
    P = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), n)) ** 2
    band = (f >= 0.5) & (f <= 8)
    f0 = hr / 60 if not np.isnan(hr) else f[band][np.argmax(P[band])]
    near = ((np.abs(f - f0) < 0.2) | (np.abs(f - 2 * f0) < 0.2)) & band
    snr = P[near].sum() / (P[band].sum() + 1e-12)
    ibi = np.diff(peaks) / FS_BVP
    ibi = ibi[(ibi > 0.3) & (ibi < 1.5)]
    ibi_cv = ibi.std() / ibi.mean() if len(ibi) >= 3 else np.nan
    pre, post = int(0.25 * FS_BVP), int(0.45 * FS_BVP)
    beats = [x[p - pre:p + post] for p in peaks if p - pre >= 0 and p + post <= len(x)]
    if len(beats) >= 3:
        M = np.array([(b - b.mean()) / (b.std() + 1e-9) for b in beats])
        t = M.mean(0)
        tmpl = float(np.mean([np.corrcoef(m, t)[0, 1] for m in M]))
    else:
        tmpl = 0.0
    mag = np.linalg.norm(acc, axis=1)
    return dict(snr=snr, skew=skew(x), kurt=kurtosis(x), ibi_cv=ibi_cv, tmpl_corr=tmpl,
                acc_std=mag.std(), acc_range=np.ptp(mag), bvp_std=x.std())


def process_subject(sub):
    pk_ecg = ecg_peaks(sub["ecg"])
    bvp = bandpass(sub["bvp"], 0.5, 8, FS_BVP, 3)
    T = int(min(len(bvp) / FS_BVP, len(sub["acc"]) / FS_ACC, len(sub["ecg"]) / FS_ECG))
    rows = []
    for t0 in range(0, T - WIN_S + 1, STEP_S):
        t1 = t0 + WIN_S
        xw = bvp[t0 * FS_BVP:t1 * FS_BVP]
        aw = sub["acc"][t0 * FS_ACC:t1 * FS_ACC]
        pk = ppg_peaks(xw)
        hr_ppg = hr_from_ibis(np.diff(pk) / FS_BVP)
        lab = sub["label"][t0 * FS_ECG:t1 * FS_ECG].astype(int)
        r = dict(subject=sub["subject"], t0=t0, hr_ecg=hr_ecg_window(pk_ecg, t0, t1), hr_ppg=hr_ppg,
                 label=np.bincount(lab).argmax() if len(lab) else 0)
        r.update(quality_features(xw, pk, hr_ppg, aw))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["err"] = (df.hr_ppg - df.hr_ecg).abs()
    # usable = PPG estimate exists, ECG reference exists, and they agree. NaN where no reference/estimate.
    df["usable"] = np.where(df.err.notna(), (df.err <= TOL_BPM).astype(float), np.nan)
    return df


def mae_vs_coverage(err, score, fracs=np.linspace(0.1, 1.0, 10)):
    e = np.asarray(err)[np.argsort(-np.asarray(score))]          # most trusted first
    return np.array([e[:max(1, int(round(f * len(e))))].mean() for f in fracs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.root, "S*", "S*.pkl")))
    assert paths, f"no S*/S*.pkl found under {a.root}"
    dfs = []
    for p in paths:
        d = process_subject(load_subject(p))
        print(f"{d.subject.iloc[0]}: {len(d)} windows, usable={np.nanmean(d.usable):.2f}, "
              f"no-estimate={d.err.isna().mean():.2f}")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(a.out, "windows.csv"), index=False)

    ev = df[df.err.notna()].copy()
    rng = np.random.default_rng(0)
    scores = {"random": rng.random(len(ev)), "accel (low motion)": -ev.acc_std.values,
              "SNR": ev.snr.values, "template corr": ev.tmpl_corr.values,
              "IBI regularity": -ev.ibi_cv.fillna(1.0).values}
    fr = np.linspace(0.1, 1.0, 10)
    curves = {k: mae_vs_coverage(ev.err.values, v, fr) for k, v in scores.items()}
    pd.DataFrame(curves, index=fr).rename_axis("coverage").to_csv(os.path.join(a.out, "baseline_mae_vs_coverage.csv"))
    print(f"\nwindows with estimate+reference: {len(ev)}  | ungated MAE = {ev.err.mean():.2f} bpm "
          f"| usable rate = {ev.usable.mean():.2f}")
    print("\nbaseline gate        AUROC(usable)   MAE@50%   MAE@100%")
    for k, v in scores.items():
        auc = roc_auc_score(ev.usable, v) if ev.usable.nunique() > 1 else float("nan")
        print(f"{k:<20} {auc:>10.3f} {curves[k][4]:>10.2f} {curves[k][-1]:>10.2f}")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 4))
    for k, c in curves.items():
        plt.plot(fr * 100, c, marker="o", ms=3, label=k)
    plt.xlabel("Coverage kept (%)"); plt.ylabel("HR MAE (bpm)"); plt.title("Baseline gates: MAE vs. coverage")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.out, "baseline_mae_vs_coverage.png"), dpi=150)


if __name__ == "__main__":
    main()
