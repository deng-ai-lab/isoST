#!/usr/bin/env python3
"""Map published MERFISH CCF coordinates to the isoST CCFv3 feature volume.

This samples an existing feature volume; it does not perform image registration
or create the five image channels from a raw CCF template.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def voxel_indices(coordinates: np.ndarray, spacing_mm: float, downrate: int) -> np.ndarray:
    """Convert CCF (x, y, z) millimetres to volume (z, y, x) indices."""
    return np.rint(coordinates[:, [2, 1, 0]] / (spacing_mm * downrate)).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", required=True, type=Path,
                        help="Allen Zhuang-ABCA-*-CCF ccf_coordinates.csv")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV or CSV.gz with cell labels and sampled features")
    parser.add_argument("--volume", type=Path,
                        default=Path("data/CCFv3_feature/volume_downx2_features.pt"))
    parser.add_argument("--spacing-mm", type=float, default=0.01,
                        help="Native CCF voxel spacing in millimetres (default: 0.01)")
    parser.add_argument("--downrate", type=int, default=2,
                        help="Feature-volume downsampling factor (default: 2)")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    if args.spacing_mm <= 0 or not np.isfinite(args.spacing_mm):
        parser.error("--spacing-mm must be finite and positive")
    if args.downrate <= 0 or args.chunk_size <= 0:
        parser.error("--downrate and --chunk-size must be positive")
    volume = torch.load(args.volume, map_location="cpu")
    if not isinstance(volume, torch.Tensor) or volume.ndim != 4:
        raise ValueError("Expected a (z, y, x, features) PyTorch tensor")
    feature_names = [f"image_feature_{i}" for i in range(volume.shape[-1])]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = in_bounds_count = 0
    for chunk in pd.read_csv(args.coordinates, dtype={"cell_label": str},
                             usecols=["cell_label", "x", "y", "z"],
                             chunksize=args.chunk_size):
        coordinates = chunk[["x", "y", "z"]].to_numpy(dtype=np.float64)
        finite = np.isfinite(coordinates).all(axis=1)
        indices = np.full((len(chunk), 3), -1, dtype=np.int64)
        indices[finite] = voxel_indices(coordinates[finite], args.spacing_mm, args.downrate)
        valid = finite & (indices >= 0).all(axis=1)
        valid &= (indices < np.array(volume.shape[:3])).all(axis=1)
        features = np.full((len(chunk), len(feature_names)), np.nan, dtype=np.float32)
        if valid.any():
            z, y, x = indices[valid].T
            features[valid] = volume[z, y, x].numpy()

        result = chunk.rename(columns={"x": "x_ccf", "y": "y_ccf", "z": "z_ccf"}).copy()
        result[["voxel_z", "voxel_y", "voxel_x"]] = indices
        result["in_bounds"] = valid
        result[feature_names] = features
        result.to_csv(args.output, mode="w" if total == 0 else "a",
                      header=total == 0, index=False)
        total += len(chunk)
        in_bounds_count += int(valid.sum())
    if total == 0:
        raise ValueError("Coordinate file has no rows")
    print(f"Mapped {in_bounds_count}/{total} cells to {args.volume}; wrote {args.output}")


if __name__ == "__main__":
    main()
