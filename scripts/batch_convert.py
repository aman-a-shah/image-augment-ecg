#!/usr/bin/env python3
"""Folder in -> folder out: convert every ECG file into a smartphone photo.

Drop ECG files (XML, WFDB .hea, or clean ECG images) into one folder, run this,
and get smartphone-photo versions in another folder.

    python scripts/batch_convert.py --in input_ecgs --out output_photos
    python scripts/batch_convert.py --in input_ecgs --out output_photos --n 3 --metadata --compare

  --n         photos per input file (default 1)
  --metadata  for XML/WFDB inputs, also write metadata.json + warp.npy (ground truth)
  --compare   also write a before/after side-by-side PNG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from physiorender.assemble import build_metadata  # noqa: E402
from physiorender.degrade import DegradationEngine  # noqa: E402
from physiorender.ingest import load_ecg  # noqa: E402
from physiorender.render import ECGRenderer  # noqa: E402
from physiorender.sampling import ParameterSampler  # noqa: E402

SIGNAL_EXTS = {".xml", ".hea"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_PAGE_W_MM = 250.0


def _compare(before: Image.Image, after: Image.Image, path: Path) -> None:
    h = min(before.height, after.height, 800)
    bw, aw = int(before.width * h / before.height), int(after.width * h / after.height)
    canvas = Image.new("RGB", (bw + aw + 16, h), (255, 255, 255))
    canvas.paste(before.resize((bw, h)).convert("RGB"), (0, 0))
    canvas.paste(after.resize((aw, h)).convert("RGB"), (bw + 16, 0))
    d = ImageDraw.Draw(canvas)
    for x, t in ((0, "BEFORE"), (bw + 16, "AFTER")):
        d.rectangle([x, 0, x + 150, 40], fill=(0, 0, 0))
        d.text((x + 10, 10), t, fill=(255, 255, 255))
    canvas.save(path)


def _convert_signal(path: Path, out: Path, *, n, seed, dpi, sampler, engine,
                    metadata, compare) -> int:
    record = load_ecg(path, validate=True)
    render = ECGRenderer(dpi=dpi).render(record)
    made = 0
    for i in range(n):
        s = seed + i
        params = sampler.sample(np.random.default_rng(s))
        result = engine.augment(render.image, params, seed=s,
                                lead_bboxes=render.lead_bboxes)
        suffix = "" if n == 1 else f"_{s:03d}"
        jpg = out / f"{path.stem}_photo{suffix}.jpg"
        result.image.save(jpg, quality=92)
        if metadata:
            warp = f"{path.stem}_photo{suffix}.warp.npy"
            np.save(out / warp, result.warp_field_inverse())
            meta = build_metadata(record, render, result,
                                  image_id=f"{path.stem}{suffix}", warp_field_filename=warp)
            meta.save(str(out / f"{path.stem}_photo{suffix}.json"))
        if compare:
            _compare(render.image, result.image, out / f"{path.stem}_compare{suffix}.png")
        made += 1
    return made


def _convert_image(path: Path, out: Path, *, n, seed, sampler, compare) -> int:
    clean = Image.open(path).convert("RGB")
    dpi = max(72, int(round(clean.width / (_PAGE_W_MM / 25.4))))
    engine = DegradationEngine(dpi=dpi)
    made = 0
    for i in range(n):
        s = seed + i
        params = sampler.sample(np.random.default_rng(s))
        result = engine.augment(clean, params, seed=s, lead_bboxes=None)
        suffix = "" if n == 1 else f"_{s:03d}"
        result.image.save(out / f"{path.stem}_photo{suffix}.jpg", quality=92)
        if compare:
            _compare(clean, result.image, out / f"{path.stem}_compare{suffix}.png")
        made += 1
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-convert a folder of ECGs to photos.")
    ap.add_argument("--in", dest="inp", default="input_ecgs")
    ap.add_argument("--out", default="output_photos")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300, help="Render DPI for XML/WFDB inputs.")
    ap.add_argument("--metadata", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    if not inp.is_dir():
        print(f"input folder not found: {inp}")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    sampler = ParameterSampler()
    engine = DegradationEngine(dpi=args.dpi)
    files = sorted(p for p in inp.iterdir() if p.is_file())
    n_ok = n_made = n_skip = n_err = 0

    for path in files:
        ext = path.suffix.lower()
        if ext == ".dat":  # WFDB data file; handled via its .hea
            continue
        try:
            if ext in SIGNAL_EXTS:
                made = _convert_signal(path, out, n=args.n, seed=args.seed, dpi=args.dpi,
                                       sampler=sampler, engine=engine,
                                       metadata=args.metadata, compare=args.compare)
                kind = "signal"
            elif ext in IMAGE_EXTS:
                made = _convert_image(path, out, n=args.n, seed=args.seed,
                                      sampler=sampler, compare=args.compare)
                kind = "image"
            else:
                print(f"  skip   {path.name} (unsupported {ext})")
                n_skip += 1
                continue
            n_ok += 1
            n_made += made
            print(f"  ok     {path.name} [{kind}] -> {made} photo(s)")
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the batch
            n_err += 1
            print(f"  ERROR  {path.name}: {exc}")

    print(f"\nDone: {n_ok} files converted -> {n_made} photos in {out}/  "
          f"({n_skip} skipped, {n_err} errors)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
