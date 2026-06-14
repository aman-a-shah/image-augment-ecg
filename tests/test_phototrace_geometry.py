"""Phase 6 gate: PhotoTrace geometry (Stage 1 corners, Stage 2 lead boxes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.slow  # trains models; deselect with -m "not slow"

from physiorender.dataset_gen import generate_dataset  # noqa: E402

from phototrace.data import BBOX_KEYS, ECGPhotoDataset  # noqa: E402
from phototrace.geometry import corners_from_h_inv, unwarp_image  # noqa: E402
from phototrace.infer import GeometryPipeline  # noqa: E402
from phototrace.models import CornerRegressor, LeadDetector  # noqa: E402
from phototrace.train_geometry import split_dataset, train_regressor  # noqa: E402

from .conftest import DUMMY_XML

IMAGE_SIZE = 96


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory) -> Path:
    if not DUMMY_XML.exists():
        pytest.skip("sample file absent")
    out = tmp_path_factory.mktemp("phototrace_ds")
    generate_dataset([DUMMY_XML], out_dir=out, n_per_source=48, dpi=100,
                     save_warp=False, write_signals=False, base_seed=100)
    return out


def test_dataset_targets_shapes(dataset_dir):
    ds = ECGPhotoDataset(dataset_dir, image_size=IMAGE_SIZE)
    sample = ds[0]
    assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["corners"].shape == (8,)
    assert sample["boxes"].shape == (len(BBOX_KEYS) * 4,)
    assert (sample["corners"] >= 0).all() and (sample["corners"] <= 1.2).all()


def test_stage1_corner_regression_learns(dataset_dir):
    train, val = split_dataset(str(dataset_dir), image_size=IMAGE_SIZE, seed=1)
    report = train_regressor(CornerRegressor(), train, val, target_key="corners",
                             epochs=40, lr=2e-3, batch_size=8, seed=1)
    # Loss must drop substantially, and beat predict-the-mean.
    assert report.train_losses[-1] < 0.6 * report.train_losses[0]
    assert report.beats_baseline, (report.val_mae, report.baseline_mae)


def test_stage2_lead_detection_learns(dataset_dir):
    train, val = split_dataset(str(dataset_dir), image_size=IMAGE_SIZE, seed=2)
    report = train_regressor(LeadDetector(), train, val, target_key="boxes",
                             epochs=40, lr=2e-3, batch_size=8, seed=2)
    assert report.train_losses[-1] < 0.6 * report.train_losses[0]
    assert report.beats_baseline, (report.val_mae, report.baseline_mae)


def test_geometry_pipeline_runs(dataset_dir):
    train, val = split_dataset(str(dataset_dir), image_size=IMAGE_SIZE, seed=3)
    c = train_regressor(CornerRegressor(), train, val, target_key="corners",
                        epochs=10, batch_size=8, seed=3).model
    l = train_regressor(LeadDetector(), train, val, target_key="boxes",
                        epochs=10, batch_size=8, seed=3).model
    pipe = GeometryPipeline(c, l, image_size=IMAGE_SIZE, out_size=(512, 360))

    items = ECGPhotoDataset(dataset_dir).items
    photo = Image.open(Path(dataset_dir) / items[0]["image"]).convert("RGB")
    out = pipe.process(photo)
    assert out.unwarped.shape == (360, 512, 3)
    assert len(out.lead_crops) >= 6  # most leads localized
    for crop in out.lead_crops.values():
        assert crop.ndim == 3 and crop.size > 0


def test_unwarp_recovers_geometry(dataset_dir):
    """Ground-truth corners unwarp the photo back to a flat doc (sanity, no model)."""
    items = ECGPhotoDataset(dataset_dir).items
    from physiorender.metadata import ECGMetadata
    meta = ECGMetadata.load(str(Path(dataset_dir) / items[0]["metadata"]))
    photo = np.asarray(Image.open(Path(dataset_dir) / items[0]["image"]).convert("RGB"),
                       np.float32) / 255.0
    h, w = photo.shape[:2]
    corners = corners_from_h_inv(np.array(meta.homography_inv), w, h)
    unwarped = unwarp_image(photo, corners, (w, h))
    assert unwarped.shape == photo.shape
