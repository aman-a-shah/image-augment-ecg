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

**Phase 0 — Foundation & contracts: complete.** The two core data contracts are
in place and tested:

- `physiorender.AugmentationParams` — the degradation parameter set (plan §7)
- `physiorender.ECGMetadata` — the per-image metadata record (plan §8)

Everything downstream reads/writes against these.

## Layout

```
physiorender/      # augmentation engine (renderer + degradation, Phases 2-5)
  config.py        # rendering constants (DPI, grid colors, paper specs)
  params.py        # AugmentationParams contract
  metadata.py      # ECGMetadata contract
  logging_setup.py # shared logging
phototrace/        # digitization models (Phases 6-7)
tests/             # pytest suite
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
