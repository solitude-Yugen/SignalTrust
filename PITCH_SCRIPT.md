# SignalTrust — 3-Minute Pitch Script
*Navigate deck with ← → arrow keys or spacebar*

---

## Slide 1 — Title (15 s)

> "Hi, I'm Vishruti from D.Y. Patil College of Engineering, Pune.
> My project is **SignalTrust** — quality-aware heart rate monitoring from wrist PPG.
> The core idea: don't just output a heart rate — first decide whether you *should*."

---

## Slide 2 — The Problem (25 s)

> "Every consumer wearable reports HR continuously — even when the PPG signal is completely corrupt from motion.
> On the WESAD dataset with real subjects doing stress protocols, the ungated error is **8.76 bpm**.
> That's nearly clinically meaningless. And the device gives no warning.
> SignalTrust scores each window and **abstains** when the signal can't be trusted — reporting HR only when it's safe to do so."

---

## Slide 3 — The Approach (30 s)

> "The pipeline is clean and honest.
> Raw wrist BVP and accelerometer → 8 signal quality features → a calibrated HistGradientBoosting classifier trained with **Leave-One-Subject-Out** cross-validation — no data leakage.
> The model outputs a calibrated probability score per 8-second window.
> The gate threshold of **0.70** was chosen on held-out subjects to keep false-accepts under 5%.
> If quality score is above 0.70 — we report HR. Below — we abstain."

---

## Slide 4 — Results (35 s)

> "Real results. 38,879 windows. 15 subjects. Fully held-out.
> Ungated MAE: 8.76 bpm. Gated MAE at 50% coverage: **2.65 bpm** — a **70% reduction**.
> AUROC of 0.864. Calibration ECE of just 0.005 — the probability scores are trustworthy.
> False-accept rate held at **4.9%** — just under our 5% target.
> And when tested on BIDMC — a completely different clinical dataset — **AUROC 0.908** with no retraining.
> The model vs. SNR comparison: our model beats the SNR baseline by ~0.5 bpm at 50% coverage. Accelerometer alone is clearly insufficient."

---

## Slide 5 — Robustness (25 s)

> "We ran a stress test: 5 synthetic artifact types at 4 severities each.
> The gate works for most — coupled motion at severity 3 drops from 25 bpm ungated to just **3.8 bpm** gated.
> I'm going to be honest about the failure mode: **strong PPG-only periodic motion** — where the artifact mimics a real pulse — fools both our model and the SNR baseline.
> The accelerometer catches coupled motion; nobody catches PPG-only. That's the next thing to fix."

---

## Slide 6 — Error Analysis (20 s)

> "We built a full failure gallery — not just top-line numbers.
> Decision landscapes, false accept and reject breakdowns, motion decile analysis, WESAD condition breakdowns, calibration reliability diagram.
> False-accept rate is 4.9%. The cost: 30% of usable windows withheld — that's the price of the 5% cap.
> Motion is the dominant driver: usable rate drops from 85% in calm windows to 17% in the most active decile."

---

## Slide 7 — Live Dashboard (20 s)

> "Here's the live demo — Streamlit dashboard running at localhost:8501.
> Pick a subject, adjust the gate threshold slider, hit Play.
> It replays the session window-by-window — quality score timeline, signal features, HR reported or withheld, running gated vs. ungated MAE curve.
> This is a replay of pre-computed scores — I'm not claiming real-time inference."

---

## Slide 8 — Limitations & Roadmap (15 s)

> "Honest limitations: single dataset, synthetic stress test, non-causal filter.
> The roadmap fixes the failure mode with PPG–ACC spectral coherence, adds causal inference for on-device deployment, and extends evaluation to more datasets."

---

## Slide 9 — Close (15 s)

> "SignalTrust: 2.65 bpm at 50% coverage. AUROC 0.864. Cross-dataset 0.908. False-accept 4.9%.
> When the wearable doesn't know — **we make it say so**.
> Thank you."

---

*Total: ~3 min. If running long, compress slide 4 — skip the per-row table, just say the headline numbers.*
