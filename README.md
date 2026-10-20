# SignalTrust 💓

**Quality-Aware Heart Rate Monitoring from Wrist PPG**  
*AI Innovation Hackathon 2026 · Biomedical AI · Challenge 09: WearableSense AI*  
*Individual submission — Vishruti, D.Y. Patil College of Engineering, Akurdi, Pune*

> **"We don't trust the wearable. We measure when to, and prove it."**

---

## What this is

Consumer wearables report heart rate continuously — even when the PPG signal is corrupt from motion, low perfusion, or sensor contact. **SignalTrust** scores each 8 s window of wrist PPG and **abstains from reporting HR when the signal is untrustworthy**, instead of silently outputting a wrong value.

The core idea: treat signal quality assessment as a **coverage-vs-accuracy trade-off** problem. A calibrated ML gate selects the fraction of windows to output, and we show the benefit rigorously with MAE-vs-coverage curves.

---

## Key Results (WESAD, LOSO, 15 subjects, real data)

| Metric | Value |
|--------|-------|
| Ungated PPG-HR MAE (no gate) | **8.76 bpm** |
| Gated MAE @ 50% coverage | **2.65 bpm** |
| Gated MAE @ 47% coverage (threshold 0.70) | **2.40 bpm** |
| Model AUROC (quality classification) | **0.864** (σ = 0.033) |
| Calibration ECE | **0.005** |
| False-accept rate @ threshold 0.70 | **4.9%** |
| BIDMC cross-dataset AUROC (5-feat PPG-only) | **0.908** |

**The gate reduces HR error by ~70% at 50% coverage** relative to reporting always.

### MAE vs. Coverage (all windows, LOSO)

| Coverage | Model | SNR gate | Accel gate | Ungated |
|----------|-------|----------|------------|---------|
| 30% | **1.81** | 1.88 | 3.23 | 8.76 |
| 50% | **2.65** | 3.12 | 3.86 | 8.76 |
| 70% | **4.72** | 5.63 | 5.24 | 8.76 |
| 90% | 7.29 | 7.72 | 7.20 | 8.76 |

*Lower is better. Model beats SNR baseline by ~0.5 bpm at 50% coverage. Accelerometer alone is insufficient.*

---

## Method

### Pipeline overview

```
Raw WESAD pkl
    │
    ▼
Phase 1: Feature extraction (src/phase1.py)
  • 8 s windows, 2 s stride, 64 Hz BVP
  • Band-pass 0.5–8 Hz (zero-phase, offline)
  • Label: usable = |HR_ppg − HR_ecg| ≤ 5 bpm
  • Features (8): snr, skew, kurt, ibi_cv,
    tmpl_corr, acc_std, acc_range, bvp_std
    │
    ▼
Phase 2: Quality model + calibration (src/phase2.py)
  • Leave-one-subject-out (LOSO), no window-level leakage
  • HistGradientBoostingClassifier + isotonic calibration
  • Threshold 0.70 selected on held-out subjects (FA ≤ 5%)
  • Output: p_cal per window → gate decision
    │
    ▼
Gate decision: p_cal ≥ 0.70 → report HR; else abstain
```

### Features

| Feature | Description |
|---------|-------------|
| `snr` | Spectral SNR at estimated HR frequency |
| `skew` | BVP waveform skewness |
| `kurt` | Kurtosis |
| `ibi_cv` | Inter-beat-interval coefficient of variation |
| `tmpl_corr` | Beat template correlation |
| `acc_std` | Wrist accelerometer standard deviation |
| `acc_range` | Accelerometer range |
| `bvp_std` | BVP amplitude standard deviation |

### Model

- **HistGradientBoostingClassifier**: `max_iter=150, learning_rate=0.08, max_leaf_nodes=24, l2_regularization=1.0`
- **Calibration**: isotonic regression via inner GroupKFold(5) on training fold
- **Split**: LOSO — 15 folds, each subject held out once
- **Threshold**: 0.70 (chosen to cap false-accept rate ≤ 5% on held-out subjects)

---

## Robustness (Stress Test)

Synthetic artifacts injected on clean quiet windows (WESAD, 1,500 windows/condition):

| Artifact | Sev | Bad rate | Ungated MAE | Gated MAE | FA rate |
|----------|-----|----------|-------------|-----------|---------|
| Coupled motion | 2 | 49.5% | 12.9 bpm | **2.51 bpm** | 1.6% |
| Coupled motion | 3 | 71.7% | 25.4 bpm | **3.78 bpm** | 1.0% |
| Dropout | 4 | 87.5% | 7.55 bpm | **3.81 bpm** | 1.6% |
| Wander | 4 | 98.1% | 7.31 bpm | **2.66 bpm** | 0.3% |
| **PPG-only motion** | **4** | **88.5%** | **34.4 bpm** | **37.1 bpm ⚠️** | **67.8%** |

**Known failure mode**: strong periodic PPG-only motion in the pulse band (accelerometer calm). Both the model and SNR gate fail — the SNR feature is partly self-consistent with the PPG estimator's Hz estimate, so motion-frequency-locked estimates score as "clean." Coupled motion is caught because the accelerometer spike triggers rejection.

---

## Cross-Dataset (BIDMC)

5-feature PPG-only model (drops `acc_std`, `acc_range`, `bvp_std` — BIDMC has no accelerometer):

| Metric | WESAD in-domain | BIDMC cross-dataset |
|--------|-----------------|----------------------|
| AUROC | 0.839 | **0.908** |
| ECE | 0.060 | **0.089** |
| Gated MAE | 2.40 bpm | — |

BIDMC AUROC is *higher* than in-domain because BIDMC (clinical, resting) has cleaner signal than WESAD lab stress conditions. The model transfers without retraining.

---

## Failure Gallery (Step 5)

See [`outputs/failure_gallery/`](outputs/failure_gallery/):

- **Decision landscape** — scatter of all 38,879 windows colored by outcome
- **False accept analysis** — 1,904 windows passed with large error; key cause: SNR self-validation
- **False reject analysis** — 7,029 usable windows withheld; 30% cost of the 5% FA cap
- **Motion decile breakdown** — usable rate drops from 85% (calmest) to 17% (most active decile)
- **WESAD condition breakdown** — Baseline best (clean), Stress worst (motion + vasoconstriction)
- **Reliability diagram** — ECE 0.089 on BIDMC, well-calibrated
- **Gate operating curve** — shows the coverage / FA rate / MAE trade-off across all thresholds

---

## Interactive Dashboard

```bash
# Install
pip install -r requirements.txt

# Run (from project root)
streamlit run app/app.py -- --oof outputs/phase2/oof_predictions.csv

# With raw BVP waveforms (optional; requires WESAD data)
streamlit run app/app.py -- \
    --oof outputs/phase2/oof_predictions.csv \
    --wesad-root <path/to/S2..S17>
```

Features: subject picker · quality threshold slider · window-by-window replay · quality score timeline · MAE vs. coverage chart · signal feature bars · recent window log

*The dashboard replays pre-computed scores (offline). The BVP filter is non-causal (`filtfilt`); a real-time deployment would need a causal filter.*

---

## Reproduce

```bash
# 1. Feature extraction (requires raw WESAD data)
python src/phase1.py \
    --root <dir containing S2..S17> \
    --out  outputs/phase1

# 2. Model training + LOSO evaluation
python src/phase2.py \
    --windows outputs/phase1/windows.csv \
    --out     outputs/phase2

# 3. Cross-dataset evaluation (requires BIDMC)
python src/bidmc_eval.py \
    --root    <bidmc dir> \
    --windows outputs/phase1/windows.csv \
    --oof     outputs/phase2/oof_predictions.csv \
    --out     outputs/cross-dataset\ test

# 4. Robustness stress test (requires raw WESAD data)
python src/stress_test.py \
    --root    <dir containing S2..S17> \
    --windows outputs/phase1/windows.csv \
    --oof     outputs/phase2/oof_predictions.csv \
    --out     outputs/stress\ test

# 5. Failure gallery
python src/failure_gallery.py \
    --oof     outputs/phase2/oof_predictions.csv \
    --windows "outputs/phase 1/windows.csv" \
    --out     outputs/failure_gallery

# 6. Dashboard
streamlit run app/app.py -- --oof outputs/phase2/oof_predictions.csv
```

---

## Repository Structure

```
SignalTrust/
├── app/
│   └── app.py                      # Streamlit dashboard (Step 4)
├── src/
│   ├── phase1.py                   # Feature extraction
│   ├── phase2.py                   # Model training + LOSO eval
│   ├── bidmc_eval.py               # Cross-dataset evaluation (Step 3)
│   ├── stress_test.py              # Robustness stress test (Step 2)
│   └── failure_gallery.py          # Failure analysis (Step 5)
├── outputs/
│   ├── phase1/                     # windows.csv, baseline figures
│   ├── phase2/                     # oof_predictions.csv, metrics
│   ├── cross-dataset test/         # BIDMC results
│   ├── stress test/                # Robustness results
│   └── failure_gallery/            # 8 analysis figures + error_analysis.md
├── requirements.txt
├── README.md
└── HANDOVER.md
```

---

## Limitations

1. **Single dataset**: WESAD, 15 subjects, one wrist device (Empatica E4), lab conditions.
2. **Label noise**: ~15% of quiet-wrist windows still have PPG–ECG disagreement; cause unverified (low perfusion, non-wrist motion, ECG detector noise).
3. **Non-causal filter**: `filtfilt` used throughout; streaming mode replays pre-computed scores.
4. **SNR self-consistency**: the `snr` feature places f₀ at the estimated HR; it cannot detect PPG-only periodic motion in the pulse band.
5. **Synthetic stress test**: artifact injection on clean quiet windows — not a population-level rate estimate.
6. **15-subject variance**: per-subject AUROC range 0.83–0.92; reported metrics have wide uncertainty intervals.
7. **Research prototype only**: no clinical or diagnostic claims; public datasets only.

---

## Roadmap

- Independent spectral quality feature (PPG–ACC coherence) to close the PPG-only motion gap
- Causal on-device inference pipeline
- Artifact-type classification (motion / dropout / saturation / wander)
- PPG/ECG fusion for clinical-grade accuracy
- Evaluation on PPG-DaLiA, larger cohorts, consumer wearable exports
- Learned end-to-end quality score (1D CNN or transformer)

---

## Data

- **WESAD**: Schmidt et al., 2018. Wearable Stress and Affect Detection. ACM ICMI. [Link](https://dl.acm.org/doi/10.1145/3242969.3242985)
- **BIDMC PPG and Respiration Dataset**: Pimentel et al., PhysioNet. [Link](https://physionet.org/content/bidmc/1.0.0/)

---

## License

MIT License — see [LICENSE](LICENSE)

*Research/educational prototype. Not for clinical use.*
