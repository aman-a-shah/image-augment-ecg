# ECG Photo Realism Pipeline

Turn clean ECGs into realistic smartphone photos — and recover the signal back.

- **PhysioRender** (`physiorender/`) — a physically-grounded augmentation engine
  that renders clean ECG signal data as realistic smartphone photos (paper
  texture, wrinkles, perspective, lighting, blur, JPEG…), with full, reversible
  metadata export (ground-truth signal, lead boxes, inverse homography).
- **PhotoTrace** (`phototrace/`) — a digitization model that recovers clean
  waveform signal from such a photo.

See `plan.md` for the full PRD, `roadmap.md` for the phased build plan, and
`docs/ARCHITECTURE.md` for the system diagram.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # numpy, opencv, Pillow, scipy, wfdb, torch, pytest …
```

---

## Quick start — folder in, photos out

Drop ECG files into one folder, run one command, get smartphone-photo versions in
another. Inputs can be **XML**, **WFDB** (`.hea`/`.dat`), or **clean ECG images**
(`.png`/`.jpg`/…). Mixed folders are fine.

```bash
# 1. (optional) create sample inputs to try it out
python scripts/make_test_inputs.py --out input_ecgs        # writes 3 XMLs + 2 images

# 2. convert every file in the folder
python scripts/batch_convert.py --in input_ecgs --out output_photos
```

Result: `output_photos/<name>_photo.jpg` for each input.

| Folder | Contents |
|---|---|
| `input_ecgs/` | your ECG files (XML / WFDB / images) |
| `output_photos/` | generated smartphone photos (`*_photo.jpg`) |

Both folders are gitignored. Point `--in`/`--out` at any paths you like
(e.g. `--in ~/Desktop/my_ecgs --out ~/Desktop/my_photos`).

### Batch options

```bash
# 3 different photo variants per ECG
python scripts/batch_convert.py --in input_ecgs --out output_photos --n 3

# also write before/after side-by-side PNGs (*_compare.png)
python scripts/batch_convert.py --in input_ecgs --out output_photos --compare

# also write ground-truth labels for XML/WFDB inputs (*.json + *.warp.npy)
python scripts/batch_convert.py --in input_ecgs --out output_photos --metadata
```

> **Labels:** XML/WFDB inputs can emit full ground truth (`--metadata`): the
> waveform, per-lead bounding boxes, and the inverse homography — this is what
> makes the output usable as digitization training data. **Image inputs get the
> photo look but no labels**, since the underlying signal isn't known from a
> picture.

---

## Other handy commands

```bash
# Single ECG -> N augmented photos (+ metadata + warp field)
python scripts/augment.py            data/dummy-ecg.xml --n 3 --bbox

# Before/after side-by-side from an XML
python scripts/before_after.py       data/dummy-ecg.xml --n 2

# Turn ONE existing ECG image into a photo (no labels)
python scripts/augment_image.py      my_ecg.png --n 3 --compare

# Inspect / render / dump each degradation layer
python scripts/inspect_ecg.py        data/dummy-ecg.xml          # parse + validate + plot
python scripts/render_ecg.py         data/dummy-ecg.xml --bbox   # clean printout
python scripts/visual_test_degrade.py data/dummy-ecg.xml         # each effect in isolation
```

`physiocam` CLI (same as several scripts above):

```bash
python -m physiorender.cli inspect  data/dummy-ecg.xml
python -m physiorender.cli render   data/dummy-ecg.xml
python -m physiorender.cli augment  data/dummy-ecg.xml --n 3
python -m physiorender.cli generate data/dummy-ecg.xml --n 200 --out artifacts/train
```

All generated files land in `artifacts/` (gitignored). Open the PNG/JPG outputs
to see the results.

---

## PhotoTrace (photo → signal)

```bash
# 1. build a training dataset of augmented photos + labels
python -m physiorender.cli generate data/dummy-ecg.xml --n 200 --out artifacts/train

# 2. train the geometry stages + digitizer (CPU-friendly defaults; scale up on GPU)
python scripts/train_phototrace.py --data artifacts/train --digitizer-source data/dummy-ecg.xml
```

---

## Testing

```bash
pytest -m "not slow"     # fast suite (~12s, no model training) — use this normally
pytest -m "not slow" -v  # show each test name as it runs
pytest                   # everything incl. model-training tests (~5–6 min)
pytest -m slow -v        # only the model-training tests
```

The `slow` tests train real models and run silently for 1–2 min each — that's
expected, not a hang (use `-v` to watch progress).

---

## What it does (by phase)

**PhysioRender** — clean ECG → realistic photo + reversible metadata
- **Ingestion:** format-adaptive loader (GE CardioSoft XML, generic-XML & WFDB fallbacks) → validated `ECGRecord` in mV
- **Renderer:** clean 12-lead printout @300 DPI, grid + calibration pulse + per-lead boxes
- **Degradation L1–2:** paper aging (yellowing, ink density, ink skip) + handling (light-consistent wrinkles/folds, edge curl, stains/pen/fingerprint) with an invertible warp field
- **Capture L3–5:** lens distortion → perspective (exports `H_inv`) → blur → lighting/specular/banding/shadow → sensor noise → JPEG
- **Data factory:** correlation-aware `ParameterSampler`, deterministic batch generation, calibration tooling

**PhotoTrace** — photo → signal
- **Stage 1:** corner regression → perspective unwarp
- **Stage 2:** lead-box detection (fixed-layout prior)
- **Stage 3:** soft-argmax column digitizer + morphology-weighted loss; DTW / R-peak F1 / HR metrics
- **End-to-end:** `DigitizationPipeline` + domain-gap evaluation harness

Outstanding items need *real printed-and-photographed ECGs* (calibration,
holdout eval, fine-tuning) or the project's *external digital ECG model* (the
headline domain-gap number). The tooling for both is in place — see `roadmap.md`.

---

## Layout

```
physiorender/
  config.py        # rendering constants (DPI, grid colors, paper specs)
  params.py        # AugmentationParams contract (plan §7)
  metadata.py      # ECGMetadata contract (plan §8)
  ingest/          # format detection, extractors, validation, load_ecg()
  render/          # layout templates + ECGRenderer
  degrade/         # noise, warp, light, layer1-5, DegradationEngine
  sampling.py      # ParameterSampler (correlation-aware)
  dataset_gen.py   # batch generation + integrity validation
  calibration.py   # synthetic-vs-real image-stat comparison
  assemble.py      # build_metadata() from pipeline outputs
  cli.py           # `physiocam` CLI
phototrace/        # data, geometry, models, train_*, pipeline, domain_gap (Phases 6-8)
scripts/           # batch_convert, make_test_inputs, augment(_image), before_after,
                   #   render_ecg, inspect_ecg, visual_test_degrade, train_phototrace
tests/             # pytest suite (114 tests; 9 marked `slow`)
data/              # source ECG files (gitignored; ships with dummy-ecg.xml)
input_ecgs/        # your batch inputs (gitignored)
output_photos/     # batch outputs (gitignored)
artifacts/         # generated images & datasets (gitignored)
```
