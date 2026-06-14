"""`physiocam` command-line interface (plan §12 Phase 7).

Subcommands:
  inspect   <file>                 detect format, extract, validate, plot
  render    <file>                 render a clean ECG printout
  augment   <file> --n N           generate augmented photos + metadata
  generate  <files...> --n N       batch dataset generation + integrity pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .assemble import build_metadata
from .degrade import DegradationEngine
from .ingest import detect_ecg_xml_format, load_ecg, validate_record
from .render import ECGRenderer
from .sampling import ParameterSampler


def _cmd_inspect(args: argparse.Namespace) -> int:
    fmt = detect_ecg_xml_format(args.path)
    rec = load_ecg(args.path, validate=False)
    report = validate_record(rec)
    print(f"format={fmt.value} leads={rec.n_leads} fs={rec.sample_rate_hz}Hz "
          f"dur={rec.duration_s:.1f}s valid={report.ok}")
    for w in report.warnings:
        print(f"  WARN {w}")
    return 0 if report.ok else 1


def _cmd_render(args: argparse.Namespace) -> int:
    rec = load_ecg(args.path, validate=True)
    res = ECGRenderer(dpi=args.dpi, paper_speed_mm_s=args.speed,
                      gain_mm_mv=args.gain).render(rec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.path).stem}_clean.png"
    res.save(path)
    print(f"wrote {path} ({res.image.size[0]}x{res.image.size[1]})")
    return 0


def _cmd_augment(args: argparse.Namespace) -> int:
    rec = load_ecg(args.path, validate=True)
    render = ECGRenderer(dpi=args.dpi, paper_speed_mm_s=args.speed,
                         gain_mm_mv=args.gain).render(rec)
    engine = DegradationEngine(dpi=args.dpi)
    sampler = ParameterSampler()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.path).stem
    for i in range(args.n):
        seed = args.seed + i
        params = sampler.sample(np.random.default_rng(seed))
        result = engine.augment(render.image, params, seed=seed,
                                lead_bboxes=render.lead_bboxes)
        image_id = f"{stem}_aug_{seed:06d}"
        result.image.save(out / f"{image_id}.jpg", quality=92)
        warp = f"warp_{image_id}.npy"
        np.save(out / warp, result.warp_field_inverse())
        meta = build_metadata(rec, render, result, image_id=image_id,
                              warp_field_filename=warp)
        meta.save(str(out / f"{image_id}.json"))
        print(f"[{i+1}/{args.n}] {image_id}.jpg valid={meta.is_valid()}")
    print(f"outputs in {out}/")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .dataset_gen import generate_dataset, validate_dataset
    report = generate_dataset(args.sources, out_dir=args.out, n_per_source=args.n,
                              dpi=args.dpi, base_seed=args.seed,
                              write_signals=not args.no_signals)
    integ = validate_dataset(args.out)
    print(f"generated={report.n_images} valid={report.n_valid} "
          f"integrity={integ.n_valid}/{integ.n_images}")
    return 0 if integ.ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="physiocam",
                                 description="PhysioRender ECG photo simulator.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dpi", type=int, default=300)
        p.add_argument("--speed", type=int, default=25)
        p.add_argument("--gain", type=int, default=10)

    pi = sub.add_parser("inspect"); pi.add_argument("path"); pi.set_defaults(fn=_cmd_inspect)

    pr = sub.add_parser("render"); pr.add_argument("path"); common(pr)
    pr.add_argument("--out", default="artifacts/render"); pr.set_defaults(fn=_cmd_render)

    pa = sub.add_parser("augment"); pa.add_argument("path"); common(pa)
    pa.add_argument("--n", type=int, default=1); pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--out", default="artifacts/dataset"); pa.set_defaults(fn=_cmd_augment)

    pg = sub.add_parser("generate"); pg.add_argument("sources", nargs="+")
    pg.add_argument("--n", type=int, default=10); pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--dpi", type=int, default=300)
    pg.add_argument("--out", default="artifacts/train")
    pg.add_argument("--no-signals", action="store_true"); pg.set_defaults(fn=_cmd_generate)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
