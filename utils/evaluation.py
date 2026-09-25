"""Evaluation artifacts for held-out isoST interpolation sections."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def save_target_qc_artifacts(
        raw_result_dir,
        total_data_dir,
        plot_dir,
        table_dir,
):
    """Save target-level tables and plots without changing model inference.

    Predictions have already been generated without held-out tensors. Ground
    truth is loaded here only for post-hoc evaluation and visualization.
    """
    raw_result_dir = Path(raw_result_dir)
    total_data_dir = Path(total_data_dir)
    plot_dir = Path(plot_dir)
    table_dir = Path(table_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = raw_result_dir / "target_depth_mapping.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Target-to-output mapping is missing: {mapping_path}"
        )

    mapping = pd.read_csv(mapping_path).sort_values("target_nominal_z")
    metric_rows = []
    feature_rows = []

    for row in mapping.itertuples(index=False):
        prediction_path = raw_result_dir / row.relative_path
        truth_path = total_data_dir / f"shuffled_{row.target_slide_name}.pt"
        prediction = np.load(prediction_path)
        truth = torch.load(truth_path, map_location="cpu").numpy()

        if prediction.ndim != 2 or truth.ndim != 2:
            raise ValueError(
                f"Expected 2-D arrays for {row.target_slide_name}; got "
                f"prediction={prediction.shape}, truth={truth.shape}."
            )
        if prediction.shape[1] != truth.shape[1]:
            raise ValueError(
                f"Feature width mismatch for {row.target_slide_name}: "
                f"prediction={prediction.shape[1]}, truth={truth.shape[1]}."
            )

        pred_mean = prediction.mean(axis=0)
        true_mean = truth.mean(axis=0)
        pred_std = prediction.std(axis=0)
        true_std = truth.std(axis=0)
        pc_slice = slice(3, prediction.shape[1])

        metric_rows.append({
            "target_slide_name": row.target_slide_name,
            "target_nominal_z": float(row.target_nominal_z),
            "target_model_z": float(row.target_model_z),
            "relative_path": row.relative_path,
            "n_prediction_cells": int(prediction.shape[0]),
            "n_truth_cells": int(truth.shape[0]),
            "prediction_to_truth_cell_ratio": float(
                prediction.shape[0] / truth.shape[0]
            ),
            "prediction_mean_z": float(pred_mean[2]),
            "truth_mean_z": float(true_mean[2]),
            "absolute_mean_z_error": float(abs(pred_mean[2] - true_mean[2])),
            "xy_centroid_error": float(
                np.linalg.norm(pred_mean[:2] - true_mean[:2])
            ),
            "xyz_centroid_error": float(
                np.linalg.norm(pred_mean[:3] - true_mean[:3])
            ),
            "pc_mean_rmse": float(
                np.sqrt(np.mean((pred_mean[pc_slice] - true_mean[pc_slice]) ** 2))
            ),
            "pc_std_rmse": float(
                np.sqrt(np.mean((pred_std[pc_slice] - true_std[pc_slice]) ** 2))
            ),
        })

        for pc_index in range(3, prediction.shape[1]):
            feature_rows.append({
                "target_slide_name": row.target_slide_name,
                "target_nominal_z": float(row.target_nominal_z),
                "pc": f"PC{pc_index - 2}",
                "prediction_mean": float(pred_mean[pc_index]),
                "truth_mean": float(true_mean[pc_index]),
                "mean_error": float(pred_mean[pc_index] - true_mean[pc_index]),
                "prediction_std": float(pred_std[pc_index]),
                "truth_std": float(true_std[pc_index]),
                "std_error": float(pred_std[pc_index] - true_std[pc_index]),
            })

    metrics = pd.DataFrame(metric_rows).sort_values("target_nominal_z")
    features = pd.DataFrame(feature_rows)
    metrics_path = table_dir / "target_qc_metrics.csv"
    features_path = table_dir / "target_pc_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    features.to_csv(features_path, index=False)

    x = metrics["target_nominal_z"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    axes[0, 0].plot(x, metrics["truth_mean_z"], "o-", label="truth")
    axes[0, 0].plot(x, metrics["prediction_mean_z"], "x--", label="prediction")
    axes[0, 0].set_ylabel("mean model z")
    axes[0, 0].legend()
    axes[0, 1].plot(x, metrics["xy_centroid_error"], "o-")
    axes[0, 1].set_ylabel("XY centroid error")
    axes[1, 0].plot(x, metrics["pc_mean_rmse"], "o-")
    axes[1, 0].set_ylabel("PC mean RMSE")
    axes[1, 0].set_xlabel("nominal z (mm)")
    axes[1, 1].plot(x, metrics["n_truth_cells"], "o-", label="truth")
    axes[1, 1].plot(
        x,
        metrics["n_prediction_cells"],
        "x--",
        label="prediction",
    )
    axes[1, 1].set_ylabel("cell count")
    axes[1, 1].set_xlabel("nominal z (mm)")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Held-out target reconstruction QC")
    fig.tight_layout()
    profile_png = plot_dir / "target_qc_profiles.png"
    profile_pdf = plot_dir / "target_qc_profiles.pdf"
    fig.savefig(profile_png, dpi=220, bbox_inches="tight")
    fig.savefig(profile_pdf, bbox_inches="tight")
    plt.close(fig)

    heatmap = features.pivot(
        index="target_slide_name",
        columns="pc",
        values="mean_error",
    )
    ordered_pc = sorted(
        heatmap.columns,
        key=lambda name: int(name.removeprefix("PC")),
    )
    heatmap = heatmap.loc[mapping["target_slide_name"], ordered_pc]
    values = heatmap.to_numpy(dtype=float)
    vmax = float(np.nanquantile(np.abs(values), 0.98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig_height = max(4.0, 0.28 * len(heatmap) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xticks(range(len(ordered_pc)))
    ax.set_xticklabels(ordered_pc, rotation=90, fontsize=7)
    ax.set_yticks(range(len(heatmap)))
    ax.set_yticklabels([
        name.removesuffix("_log_PC") for name in heatmap.index
    ], fontsize=8)
    ax.set_xlabel("principal component")
    ax.set_ylabel("held-out target")
    ax.set_title("Prediction minus truth: PC mean error")
    fig.colorbar(image, ax=ax, label="mean error")
    fig.tight_layout()
    heatmap_png = plot_dir / "target_pc_mean_error_heatmap.png"
    heatmap_pdf = plot_dir / "target_pc_mean_error_heatmap.pdf"
    fig.savefig(heatmap_png, dpi=220, bbox_inches="tight")
    fig.savefig(heatmap_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "target_qc_metrics": str(metrics_path),
        "target_pc_summary": str(features_path),
        "target_qc_profiles_png": str(profile_png),
        "target_qc_profiles_pdf": str(profile_pdf),
        "target_pc_heatmap_png": str(heatmap_png),
        "target_pc_heatmap_pdf": str(heatmap_pdf),
    }
