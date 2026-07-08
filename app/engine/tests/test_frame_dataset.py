"""Sub-phase 1.1 verification: transforms, frame index, and the fork-safe
frame Dataset / DataLoader — validated against a real generated dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from engine.data.frame_dataset import FrameDataset, make_frame_dataloader
from engine.data.frame_index import build_frame_index
from engine.data.harness import GenerationConfig, generate_dataset
from engine.data.transforms import frame_to_tensor, tensor_to_frame
from engine.envs.grid2d.config import Grid2DConfig


@pytest.fixture
def small_dataset(tmp_path: Path) -> Path:
    """A tiny real dataset: 3 episodes x (8+1) frames at 32x32."""
    out, _ = generate_dataset(
        GenerationConfig(
            dataset_name="ds_1_1",
            output_dir=str(tmp_path / "ds"),
            num_episodes=3,
            episodes_per_shard=2,
            num_workers=1,
            env=Grid2DConfig(canvas_size=32, episode_length=8, num_obstacles=2),
        )
    )
    return out


def test_frame_to_tensor_shape_and_range() -> None:
    frame = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    t = frame_to_tensor(frame, target_size=(64, 64))
    assert t.shape == (3, 64, 64)
    assert t.dtype == torch.float32
    assert t.min() >= -1.0 - 1e-6 and t.max() <= 1.0 + 1e-6


def test_transform_round_trip_is_close() -> None:
    frame = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    # same-size round trip (no resize) should be near-lossless
    t = frame_to_tensor(frame, target_size=(64, 64))
    back = tensor_to_frame(t)
    assert back.shape == (64, 64, 3)
    assert np.abs(back.astype(int) - frame.astype(int)).max() <= 1


def test_frame_index_counts_all_frames(small_dataset: Path) -> None:
    index = build_frame_index(small_dataset)
    # 3 episodes, each episode_length=8 -> 9 frames (T+1)
    assert len(index) == 3 * 9


def test_index_on_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_frame_index(tmp_path / "nonexistent")


def test_dataset_yields_correct_tensors(small_dataset: Path) -> None:
    ds = FrameDataset(small_dataset, target_size=(64, 64))
    assert len(ds) == 27
    sample = ds[0]
    assert sample.shape == (3, 64, 64)
    assert sample.dtype == torch.float32


def test_dataloader_batches(small_dataset: Path) -> None:
    loader = make_frame_dataloader(
        small_dataset, batch_size=8, shuffle=True, num_workers=0
    )
    batch = next(iter(loader))
    assert batch.shape == (8, 3, 64, 64)


def test_dataloader_multiworker_fork_safe(small_dataset: Path) -> None:
    """The load-bearing test: >0 workers must not corrupt HDF5 handles and
    must return every frame exactly once across an epoch."""
    loader = make_frame_dataloader(
        small_dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=False
    )
    total = sum(batch.shape[0] for batch in loader)
    assert total == 27  # all frames served, no worker crash, no duplication loss