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

**Phases 0–2 complete** (41 passing tests). See `roadmap.md`.

- **Phase 0 — contracts:** `AugmentationParams` (plan §7), `ECGMetadata` (plan §8)
- **Phase 1 — ingestion:** format-adaptive loader → validated `ECGRecord` in mV
  (CardiologyXML full impl, generic heuristic + WFDB fallbacks; plan §4)
- **Phase 2 — renderer:** `ECGRecord` → photorealistic clean 12-lead printout at
  300 DPI with grid, calibration pulse, and per-lead bboxes (plan §5)

Try it:

```bash
python scripts/inspect_ecg.py data/dummy-ecg.xml          # detect + extract + plot
python scripts/render_ecg.py  data/dummy-ecg.xml --bbox   # render clean printout
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
phototrace/        # digitization models (Phases 6-7)
scripts/           # inspect_ecg.py, render_ecg.py
tests/             # pytest suite (41 tests)
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
