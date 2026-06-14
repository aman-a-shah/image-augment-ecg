# Architecture

Two-sided pipeline: **PhysioRender** simulates realistic ECG photos from clean
signal data with reversible metadata; **PhotoTrace** learns to invert that
degradation, recovering clean signal from a photo.

```
                              PhysioRender  (physiorender/)
┌───────────────────────────────────────────────────────────────────────────┐
│  ECG file (XML / WFDB)                                                       │
│        │  ingest/  detect → SignalExtractor → validate                        │
│        ▼                                                                     │
│  ECGRecord  [leads × samples] mV                                             │
│        │  render/  ECGRenderer @300DPI (grid, trace, cal pulse, bboxes)       │
│        ▼                                                                     │
│  Clean printout PNG + lead_bboxes                                            │
│        │  degrade/  DegradationEngine.augment()                              │
│        │    L1 paper: yellowing, ink density, grid bleed, ink skip            │
│        │    L2 handling: wrinkles/folds/curl (warp) + stain/pen/fingerprint   │
│        │    L3 camera: lens distort → perspective(H_inv) → blur → dirt        │
│        │    L4 lighting: ambient, specular, FL banding, hand shadow           │
│        │    L5 capture: sensor noise → white balance → JPEG                   │
│        ▼                                                                     │
│  augmented.jpg  +  metadata.json (signals, bboxes, homography_inv) + warp.npy │
└───────────────────────────────────────────────────────────────────────────┘
        │   sampling.py  ParameterSampler (correlation-aware)
        │   dataset_gen.py  batch → manifest.jsonl + integrity pass
        ▼
                               PhotoTrace  (phototrace/)
┌───────────────────────────────────────────────────────────────────────────┐
│  Smartphone photo (real or PhysioRender)                                    │
│        │  Stage 1  CornerRegressor → corners → unwarp (H_inv supervision)    │
│        ▼                                                                     │
│  Fronto-parallel document                                                   │
│        │  Stage 2  LeadDetector → 13 lead boxes  (canonical-layout prior)     │
│        ▼                                                                     │
│  Per-lead strip crops                                                       │
│        │  Stage 3  ColumnDigitizer (soft-argmax) → per-column y → mV          │
│        │  postprocess: resample, baseline-wander removal, calibration         │
│        ▼                                                                     │
│  Digitized signal  → existing digital ECG model                             │
└───────────────────────────────────────────────────────────────────────────┘
        │   domain_gap.py  digital-model error: naive vs PhotoTrace (plan §14)
```

## Key invariants

- **Reversibility.** Every geometric augmentation exports its inverse: the
  handling warp as a dense `warp_field.npy`, the perspective as `homography_inv`.
  Verified by round-trip tests (<0.02 MAE) and visual de-skew.
- **Capture physics order.** lens → perspective → blur → light → noise → JPEG.
  This is what makes the data a simulator, not a filter stack (plan §14).
- **One global light source.** Layer-2 crease shading and Layer-4 scene lighting
  share a single `LightSource`, so shadows are mutually consistent.
- **Calibrated digitization.** After unwarping to the canonical layout, the
  column→mV mapping `(a, b)` is exact, so predicted column positions become
  physical millivolts.

## Module map

| Package | Modules | Phase |
|---|---|---|
| `physiorender` | `params`, `metadata`, `config` | 0 |
| `physiorender.ingest` | `formats`, `record`, `base`, `cardiology_xml`, `generic`, `wfdb_reader`, `validate` | 1 |
| `physiorender.render` | `layout`, `renderer` | 2 |
| `physiorender.degrade` | `noise`, `warp`, `light`, `imageutil`, `layer1_paper`, `layer2_handling`, `layer3_camera`, `layer4_lighting`, `layer5_capture`, `engine` | 3–4 |
| `physiorender` | `sampling`, `dataset_gen`, `calibration`, `assemble`, `cli` | 4–5, 8 |
| `phototrace` | `data`, `geometry`, `models`, `train_geometry`, `infer` | 6 |
| `phototrace` | `digitize_data`, `losses`, `metrics`, `postprocess`, `train_digitize` | 7 |
| `phototrace` | `pipeline`, `domain_gap` | 8 |
```
