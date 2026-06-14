# ECG Photo Realism Pipeline

Two-sided pipeline for closing the domain gap between clean digital ECGs and
real-world smartphone photos of printed ECGs:

- **PhysioRender** (`physiorender/`) — a physically-grounded augmentation engine
  that renders clean ECG signal data as realistic smartphone photos, with full,
  reversible metadata export.
- **PhotoTrace** (`phototrace/`) — a digitization model that recovers clean
  waveform signal from a photo, bridging back to the digital model.

See `plan.md` for the full PRD and `roadmap.md` for the phased build plan.

## Status

**All 8 phases implemented.** PhysioRender (XML → augmented photo + metadata) is
fully functional; PhotoTrace (photo → digitized signal) trains and runs
end-to-end. The only items outstanding need *real printed-and-photographed ECGs*
or the project's *external digital ECG model* — see `roadmap.md`. Architecture:
`docs/ARCHITECTURE.md`.

**PhysioRender** (`physiorender/`)
- **Phase 0 — contracts:** `AugmentationParams` (plan §7), `ECGMetadata` (plan §8)
- **Phase 1 — ingestion:** format-adaptive loader → validated `ECGRecord` in mV
- **Phase 2 — renderer:** clean 12-lead printout @300 DPI + per-lead bboxes
- **Phase 3 — degradation L1–2:** paper aging + light-consistent handling, invertible warp
- **Phase 4 — capture L3–5:** lens/perspective(`H_inv`)/blur/lighting/noise/JPEG → JPEG + metadata + warp
- **Phase 5 — data factory:** correlation-aware `ParameterSampler`, batch generation, calibration tooling

**PhotoTrace** (`phototrace/`)
- **Phase 6 — geometry:** Stage 1 corner regression (unwarp) + Stage 2 lead detection
- **Phase 7 — digitization:** Stage 3 soft-argmax column digitizer + morphology loss + DTW/F1 metrics
- **Phase 8 — end-to-end:** `DigitizationPipeline`, domain-gap harness, `physiocam` CLI

Try it:

```bash
# PhysioRender
python -m physiorender.cli inspect  data/dummy-ecg.xml
python -m physiorender.cli render   data/dummy-ecg.xml
python -m physiorender.cli augment  data/dummy-ecg.xml --n 3
python -m physiorender.cli generate data/dummy-ecg.xml --n 200 --out artifacts/train

# PhotoTrace
python scripts/train_phototrace.py --data artifacts/train --digitizer-source data/dummy-ecg.xml
```

## Layout

```
physiorender/
  config.py        # rendering constants (DPI, grid colors, paper specs)
  params.py        # AugmentationParams contract (plan §7)
  metadata.py      # ECGMetadata contract (plan §8)
  logging_setup.py # shared logging
  ingest/          # Phase 1: format detection, extractors, validation, load_ecg()
  render/          # Phase 2: layout templates + ECGRenderer
  degrade/         # Phase 3-4: noise, warp, light, layer1-5, DegradationEngine
  assemble.py      # Phase 4: build_metadata() from pipeline outputs
phototrace/        # digitization models (Phases 6-7)
scripts/           # inspect_ecg.py, render_ecg.py, visual_test_degrade.py, augment.py
tests/             # pytest suite (83 tests)
data/              # source ECG files & datasets (gitignored)
artifacts/         # generated images & outputs (gitignored)
```

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # Phase 0 contracts are stdlib-only; dev extra = pytest
pytest
```

The Phase 0 contract modules (`params`, `metadata`, `config`) are intentionally
pure-stdlib, so the test suite runs without the heavy numpy/opencv/torch stack.
Those dependencies are declared in `pyproject.toml` and installed as later phases
need them.
