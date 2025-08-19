# PRD: Spoken-Digit Classifier (Audio → Digit)

## 1) Executive Summary
Build a lightweight prototype that listens to spoken digits (0–9) and predicts the correct number quickly and reliably using the **Free Spoken Digit Dataset (FSDD, 8 kHz)**. Optimize for simplicity, responsiveness, and demonstrable LLM-assisted development. Optional bonus: live microphone inference.

---

## 2) Goals & Success Metrics
**Primary Goals**
- ✅ **Accuracy:** ≥ **95%** overall test accuracy on a speaker-holdout split (generalizes to unseen speakers).
- ✅ **Latency:** **≤ 200 ms** from audio buffer → prediction on a typical laptop CPU (feature extraction + inference).
- ✅ **Simplicity:** Minimal deps, small model (**< 5 MB**), clean modular code.
- ✅ **LLM Collaboration Evidence:** Include prompts/screenshots or short recording (≤30 min) showing how LLMs helped.

**Secondary Goals**
- Robustness to moderate background noise; demonstrate via noise-augmented tests.
- Optional live microphone inference with real-time feedback.

**Non-Goals**
- Cloud training, large-scale hyperparameter sweeps, or mobile deployment hardening.
- Multilingual digit recognition.

---

## 3) Users & Use Cases
**Primary User:** Hiring team evaluating coding workflow + LLM collaboration and prototype quality.

**Use Cases**
1. **Batch inference:** Run predictions on provided WAV files.
2. **Interactive test (optional):** Speak digits via the microphone and see immediate predictions.
3. **Reproducible evaluation:** Train/evaluate with a fixed split and report metrics.

---

## 4) Solution Overview
**Feature-first lightweight baseline**  
- Extract **MFCC** (+ Δ, ΔΔ) and/or **log-mel** features at 8 kHz.
- Train a **small classical classifier** (e.g., **SVM / Logistic Regression** or **XGBoost**) for speed and simplicity.
- Optional alt path: **tiny 1D CNN** for end-to-end learning if time permits.

**Why MFCC + SVM?**
- Strong baseline on small speech tasks.
- Very fast inference, tiny footprint, easy to implement and tune quickly.

---

## 5) Functional Requirements
1. **Dataset ingestion**
   - Download/point to FSDD (Hugging Face).
   - Parse labels from filenames.
   - Standardize audio: mono, 8 kHz, normalized amplitude.

2. **Preprocessing / Feature Extraction**
   - Trim leading/trailing silence.
   - Extract **MFCC (e.g., 13–20 coeffs)** + deltas; mean/variance normalize.
   - Fixed-length handling (pad/truncate OR aggregate with statistics like mean/var over time).

3. **Training**
   - Train/val/test split with **speaker holdout** (e.g., leave-1-speaker-out or 70/15/15 with per-speaker partitioning).
   - Train baseline model (SVM with RBF/Linear kernel or Logistic Regression).
   - Save model + scaler/featurizer artifacts.

4. **Evaluation**
   - Report accuracy, confusion matrix, per-class precision/recall.
   - Record latency benchmarks (feature + inference) on a few samples.
   - Optional: test with synthetic noise (SNR 10–20 dB) and report accuracy delta.

5. **Inference (Batch)**
   - CLI: `python infer.py --file path/to.wav` → prints digit and confidence.
   - CLI (folder): `python infer.py --dir samples/` → CSV of filename,prediction,confidence,latency_ms.

6. **Live Microphone (Optional Bonus)**
   - Simple stream or push-to-talk capture at 8 kHz.
   - On release or buffer fullness: run feature extraction → prediction; display result + confidence + latency.

7. **Artifacts**
   - `model.pkl` (or similar), `scaler.pkl`, and feature config JSON.
   - `README.md` with approach, setup, and results.
   - Short screen recording showing LLM collaboration.

---

## 6) Non-Functional Requirements
- **Performance:** ≤200 ms per inference on CPU; memory footprint <200 MB at runtime.
- **Portability:** Python 3.10+; dependencies limited to `numpy`, `scikit-learn`, `librosa`/`torchaudio`, `sounddevice` (optional mic), `matplotlib` (plots), `joblib`.
- **Modularity:** Clear separation of data, features, model, train, eval, and inference.
- **Reproducibility:** Fixed random seeds and deterministic paths where possible.
- **Logging:** Minimal structured logs (JSON or CSV) for training stats and latency.
- **Testing:** Smoke tests for feature extraction and inference.

---

## 7) Constraints & Assumptions
- Environment: Typical laptop (no GPU assumed).
- Audio: 8 kHz WAV mono; resample as needed.
- Timebox: **2–4 hours** total build time, including experiments.

---

## 8) System Design & Architecture

**File/Module Layout**
```
fsdd_digits/
  data/
    download_fsdd.py
    dataset.py          # loading, splitting, speaker metadata
  features/
    audio_io.py         # load, resample, normalize
    preprocess.py       # trim silence, VAD-lite
    mfcc.py             # feature extraction & scaling
  models/
    classical.py        # SVM/LogReg/XGB wrapper
    tiny_cnn.py         # (optional) PyTorch model
  train.py              # trains baseline; saves artifacts
  eval.py               # metrics, confusion matrix, noise robustness
  infer.py              # batch/file inference
  mic.py                # (optional) live microphone inference
  utils/
    timing.py           # latency decorators
    io.py               # artifact save/load
  README.md
  requirements.txt
```

**Data Flow**
1. WAV → load/resample → trim silence → MFCC(+Δ,ΔΔ) → scale → fixed-length representation  
2. Features → model (SVM/LogReg) → predicted digit + confidence

---

## 9) Dataset Handling
- Use the **Free Spoken Digit Dataset (FSDD)**.
- Parse speaker IDs from filenames (e.g., `2_jackson_48.wav` → label `2`, speaker `jackson`).
- **Speaker-aware split** to measure generalization.
- Optionally augment training with:
  - Additive Gaussian or real noise samples.
  - Time shift ±50 ms.
  - Gain perturbation ±3 dB.

---

## 10) Modeling Plan

**Baseline (Primary)**
- **Features:** MFCC 20 + Δ + ΔΔ (total 60 dims), 25 ms window, 10 ms hop.
- **Aggregation:** Mean + std across time (120 dims) **OR** fixed time-frames padded/truncated then flattened.
- **Model:** **SVM (RBF)** or **Logistic Regression** with L2.
- **Why:** Strong, tiny, fast to train/infer; minimal code.

**Alternative (Optional)**
- **Tiny 1D CNN** over log-mel spectrogram.
- Use if we have spare time; compare accuracy/latency.

**Hyperparameters (quick sweep)**
- SVM C ∈ {0.1, 1, 10}, gamma ∈ {scale, 0.01}.
- LogReg C ∈ {0.5, 1, 2}.  
Pick quickest stable performer.

---

## 11) Evaluation & Reporting
- **Metrics:** Overall accuracy, per-class precision/recall, confusion matrix.
- **Robustness:** Accuracy with added noise (e.g., SNR 10/20 dB).
- **Latency:** Report mean/95th percentile for feature extraction and inference on 20 random files.
- **Model size:** Artifact size on disk.

**Acceptance**
- ≥95% accuracy (speaker-holdout), **≤200 ms** per sample end-to-end, model <5 MB.

---

## 12) UX & Interfaces

**CLI**
- `python train.py --data ./fsdd --out ./artifacts`
- `python eval.py --data ./fsdd --artifacts ./artifacts --report ./reports`
- `python infer.py --file path.wav --artifacts ./artifacts`
- `python infer.py --dir samples/ --artifacts ./artifacts --out preds.csv`

**Mic (Optional)**
- `python mic.py --artifacts ./artifacts --vad_threshold 0.02 --chunk_ms 1000`

**Outputs**
- Console print of predicted digit + confidence.
- CSV reports and PNG confusion matrix.

---

## 13) Observability
- **Training log:** CSV/JSON of epoch (if CNN) or fold, accuracy, params.
- **Latency log:** CSV for feature_time_ms, infer_time_ms, total_ms.
- **Artifacts:** `model.pkl`, `scaler.pkl`, `feature_config.json`.

---

## 14) Risks & Mitigations
- **Overfit to speakers:** Use speaker-holdout split; regularize; simple features.
- **Noisy mic input:** Add noise augmentation; basic VAD/silence trimming.
- **Sample rate mismatch:** Force resample to 8 kHz on load.
- **Latency spikes:** Pre-load model, avoid GPU warm-up; keep feature dims small.

---

## 15) Project Plan (Timeboxed)
**~25–35 min**: Data + features (loader, MFCC, trim)  
**~30–45 min**: Baseline model train + eval + logs  
**~20–30 min**: Inference CLI + latency timing  
**~15–25 min**: README + metrics plots + artifact save  
**Optional 20–30 min**: Mic input + noise tests

---

## 16) Deliverables
- **Codebase** as above (Python).
- **Artifacts**: trained model + scaler + configs.
- **Reports**: metrics (CSV/PNG), latency logs, confusion matrix.
- **README.md**: setup, approach, results, latency, LLM-collab notes.
- **Screen recording** (≤30 min) demonstrating LLM usage.

---

## 17) LLM Collaboration (What to Capture)
- Prompt snippets for:
  - MFCC config recommendations and trade-offs.
  - SVM vs LogReg choice given constraints.
  - Quick fixes for audio I/O edge cases (resampling, VAD-lite).
  - Snippets for latency timing utilities.
  - Mic input scaffolding (sounddevice, stream buffering).
- Brief reflections: how LLM sped up coding/debugging and informed decisions.

---

## 18) Acceptance Criteria (Checklist)
- [ ] Trainable on FSDD with reproducible split (speaker-aware).  
- [ ] **≥95%** test accuracy; confusion matrix included.  
- [ ] **≤200 ms** end-to-end latency on CPU (report with logs).  
- [ ] CLI inference for single file & directory.  
- [ ] Clean, modular code; artifacts saved/loaded.  
- [ ] README with setup, results, and LLM collaboration evidence.  
- [ ] (Optional) Microphone demo functioning on typical laptop.

---

## 19) Appendix: Dependencies
- `python` (3.10+), `numpy`, `scipy`, `scikit-learn`, `librosa` (or `torchaudio`), `sounddevice` (optional), `matplotlib`, `joblib`, `tqdm`.
