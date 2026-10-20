"""Write fake WESAD-format subjects (same pkl structure) to test the pipeline without real data."""
import os, pickle, sys
import numpy as np

def make(root, sid, seed, dur=600):
    r = np.random.default_rng(seed)
    def hr(t): return 70 + 8 * np.sin(2 * np.pi * t / 90 + seed)
    t7 = np.arange(dur * 700) / 700
    ph7 = np.cumsum(hr(t7) / 60 / 700)
    ecg = np.zeros(len(t7)); beats = np.where(np.diff(np.floor(ph7)) > 0)[0]
    for b in beats: ecg[max(0, b - 30):b + 30] += np.exp(-0.5 * ((np.arange(max(0, b - 30), b + 30) - b) / 7) ** 2)
    ecg += 0.02 * r.standard_normal(len(ecg))
    t64 = np.arange(dur * 64) / 64; ph = np.cumsum(hr(t64) / 60 / 64); f = ph % 1
    g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2)
    bvp = g(f, .22, .07) + .4 * g(f, .55, .09) + .02 * r.standard_normal(len(f))
    t32 = np.arange(dur * 32) / 32
    acc = 0.03 * r.standard_normal((len(t32), 3)) + np.array([0, 0, 1.0])
    for s0, s1 in [(60, 90), (200, 230), (400, 440)]:
        m64 = (t64 >= s0) & (t64 < s1); m32 = (t32 >= s0) & (t32 < s1)
        bvp[m64] += 1.0 * np.sin(2 * np.pi * 2.1 * t64[m64]) + 0.4 * r.standard_normal(m64.sum())
        acc[m32] += 0.6 * np.sin(2 * np.pi * 2.1 * t32[m32])[:, None] + 0.2 * r.standard_normal((m32.sum(), 3))
    d = {"signal": {"chest": {"ECG": ecg[:, None], "ACC": np.zeros((len(ecg), 3))},
                    "wrist": {"BVP": bvp[:, None], "ACC": acc}},
         "label": np.ones(len(ecg), dtype=int), "subject": sid}
    os.makedirs(f"{root}/{sid}", exist_ok=True)
    with open(f"{root}/{sid}/{sid}.pkl", "wb") as fh: pickle.dump(d, fh)

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "tmp_wesad"
    for i, s in enumerate(["S98", "S99"]): make(root, s, i + 1)
