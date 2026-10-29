# DLDL v2 roadmap: tearing-focused state-space disruption predictor

The current DLDL model ([`src/model/cnn.py`](../src/model/cnn.py)) is a whole-shot 1D CNN on `ipspr15v`. It sees the entire shot, including the current quench, so it can only label disruptions after the fact. It cannot warn in real time.

DLDL v2 is a **causal, per-millisecond predictor** focused on **tearing modes (TMs)** and built on a **state-space model (SSM)**. It has three goals:

- use the fewest inputs possible;
- warn of disruptions **≥250 ms** ahead;
- beat Seo et al., IJCNN 2023.

## Baseline to beat

Seo et al., "Multimodal Prediction of Tearing Instabilities in a Tokamak," IJCNN 2023, doi:[10.1109/IJCNN54540.2023.10191359](https://doi.org/10.1109/IJCNN54540.2023.10191359).

- **Data:** DIII-D shots 147000–190985 (8,505 shots, 639,555 time slices), split 7:2:1 by *sample*.
- **Inputs:** 16 quantities.
  - 5 profiles at time t on 33 ψN points: n_e, T_e, ion rotation, 1/q, pressure.
  - 11 scalars at **t+25 ms**, which are future actuator requests, so the model is not strictly causal.
- **Label ("tearability"):** the n=1 Mirnov RMS is above a threshold at t+25 ms. The threshold is not given. Slices where a mode is already present count as positives.
- **Result:** best tearability **AUC = 0.907**. The 250 ms early warning is shown on one shot only (193207).

## Decisions

- **n1rms is label-only (decided 2026-10-29).** It defines the tearing labels but is never a model input. A "+n1rms" ablation is deferred.
- **Disruptions:** train on all disruptions, but choose inputs using tearing-related ones. All-cause results are reported too.
- **Backbone: S4D** (a diagonal linear SSM) in pure PyTorch.
  - Training and streaming give exactly the same output.
  - It is cheap enough for the input search (about 300 runs).
  - It ports easily to the plasma control system.
  - A Mamba-style selective layer is kept only as an ablation.
- **Input pools:** the minimal set is chosen from signals available in real time. ZIPFIT profiles and `prmtan_neped` are offline, time-smoothed fits. They are used only by the PaperNet replica and an "offline-oracle" ablation.
- **[`all_shots.csv`](../all_shots.csv) is not a label source** (see [Known issue](#known-issue)).

## Roadmap

| # | Phase | Runs on | Deliverable | Gate to move on |
|---|---|---|---|---|
| 0 | HDF5 audit | Polaris CPU | Per-signal inventory, plus a decision note on: the tearing threshold θ, sampling rates, ECE channels, disruption-label sources, and extra signals (q95, βN, locked-mode) in `signal_data/d3d/` | Go/no-go on labels and inputs |
| 1 | Labels | Polaris CPU | Tearing-mode state and onsets from n1rms; disruption time with its source; tearing-related and phase tags | Shot 193207's timeline from the paper is reproduced (onset ≈3.25 s, lock ≈3.40 s, disruption ≈3.80 s) |
| 2 | Time-series store | Polaris CPU | Causal 1 kHz arrays, a signal registry with a denylist, and a dataset class | Causality and leakage tests pass |
| 3 | Metrics + baselines | Polaris GPU | P0–P3 metrics; persistence, PaperNet, GBDT and GRU baselines | PaperNet scores ≈0.85–0.91 under the paper's protocol, confirming our labels match theirs |
| 4 | TearSSM | Polaris GPU | S4D model on the full real-time pool, 5 seeds | Beats PaperNet on validation P1 and P2 |
| 5 | Minimal inputs | Polaris GPU sweep | Input ranking, then forward selection, then finalists; Pareto plot | Smallest input set within 1 standard error of the full pool, locked before any test evaluation |
| 6 | Real time + final evaluation | local + Polaris | Streaming parity and latency checks; single test-set evaluation; figures for the paper's shots | Claim rule met |
| side | Fix historical CNN pipeline | — | `ip`→`ipspr15v`, correct time base, training-set normalization | `all_shots.csv` recall matches the 0.972 test recall |

## Splits

Splits are by shot and chronological. They reuse the CNN's existing dev/test boundaries.

| Set | Shots | Archive shots | Use |
|---|---|---|---|
| Train | 147000–172246 | 15,629 | fit |
| Val | 172247–176422 | 2,879 | early stopping, input selection, calibration |
| Test-A | 176423–180847 | 2,578 | all metrics (curated disruption labels) |
| Test-B | 180848–190985 | 5,738 | tearing metrics; disruption metrics only where labels pass validation |
| Extended | 190986–205775 | 9,577 | drift check; contains the paper's shots 193207/8/10/11 |

## Model map

```
 HDF5 archive ──► causal 1 kHz resampling ──► x (B, L, C) + availability mask
   │                                   (C = minimal set from ~18 real-time keys)
   └─ n1rms ──► labels only: tm_raw, tm_state, onsets, TM-related tag
 curated ips_labels.txt ──► t_D

 TearSSM
 x_t ⊕ mask_t
   │ per-input affine · learned "missing" embedding · signal dropout p≈0.1
   ▼
 Linear C→d  (d = 64–128)                                   (B, L, d)
 ┌──────────────── × 4–6 blocks ─────────────┐
 │ RMSNorm                                   │
 │ S4D mixer, per channel, N = 64            │   s_t = Ā s_{t−1} + B̄ u_t
 │   training: causal FFT conv, padded 2L    │   y_t = Re(C s_t) + D u_t
 │   real time: exact recurrent step()       │   Ā = exp(ΔA), Re A < 0, τ ≈ 2 ms–2 s
 │ GLU → dropout → + residual                │
 └───────────────────────────────────────────┘
   ▼ h_t                                                    (B, L, d)
   ├─► Paper head        σ → P[tm_raw(t+25 ms)]          ← compared with AUC 0.907
   ├─► Onset hazard      F_k, H = 25…1000 ms  (loss only where no mode)
   ├─► Disruption hazard F_k, H = 50…2000 ms
   └─► Aux (optional)    βp, log n1rms at t+25 ms (targets only)
        F_k = 1 − exp(−Σ_{j≤k} softplus z_j)  → monotone across horizons
   ▼
 Alarm: P(disrupt ≤ H*) > θ for k ms  →  true positive if ≥ 250 ms before t_D
 Size ≈ 100–300k params · streaming target < 100 µs/step on CPU
```

| Model | Inputs | What it answers |
|---|---|---|
| Persistence | n1rms(t) | Upper reference for the all-slices protocol, since it uses the label's own signal |
| PaperNet | Paper's 16 inputs mapped to archive stand-ins (time t, plus a t+25 ms variant) | Like-for-like comparison with the paper; P0 label check |
| GBDT | Causal window features | Strong non-deep baseline, in the style of Fu et al. 2020 |
| GRU | Same inputs as TearSSM | Does the SSM beat a recurrent network? |
| **TearSSM (S4D)** | Minimal real-time set | Primary model |
| Selective-SSM ablation | Same | Does input-dependent memory help? |
| Offline-oracle | Plus ZIPFIT profiles | How much offline profiles add (not deployable in real time) |
| +n1rms (deferred) | Plus n1rms | How much direct MHD amplitude adds |

## Evaluation protocols

- **P0, label check:** PaperNet with the paper's label, run on both a slice-random 7:2:1 split and a shot-wise split. If it doesn't land near 0.85–0.91, our threshold or label differs from theirs and must be fixed first.
- **P1, headline:** `tm_raw(t+25 ms)` on 25 ms slices of Test-A+B. ROC-AUC over all slices and over flattop only.
- **P2, onset only:** only slices with no mode at t, horizons 25/100/250/500 ms, ROC-AUC and PR-AUC. Also report the percentage of onsets warned ≥250 ms ahead, at a fixed false-alarm rate.
- **P3, disruption per shot:**
  - A true positive is a first alarm ≥250 ms before t_D. A false positive is any alarm on a non-disruptive shot.
  - Report ROC/AUC, the true-positive rate at 1% and 5% false-positive rate, and the warning-time CDF.
  - Break results down by tearing-related vs. other disruptions, and by phase.
- **Claim rule:** all of the following must hold.
  - P1 AUC > 0.907.
  - Better than PaperNet on both P1 and P2, with paired shot-bootstrap 95% confidence intervals that exclude 0.
  - Fewer than 16 inputs, all from the real-time pool.

## Known issue

On the 35,459 shots it shares with [`data/ips_labels.txt`](../data/ips_labels.txt), `all_shots.csv` (from [`src/scrape_data.py`](../src/scrape_data.py)) has recall 0.418 and precision 0.964. The same CNN reaches 0.972 recall on the `.txt` test set. The gap comes from three mismatches between training and inference:

- [`read_h5_shot`](../src/util/historical.py) reads `ip`, not `ipspr15v`.
- It builds a uniform 4 kHz grid starting at the first HDF5 sample, whereas the `.txt` training files start at −5 s with 10 ms steps.
- [`src/preprocess_historical.py`](../src/preprocess_historical.py) normalizes with statistics from the whole archive instead of the training mean/std.
