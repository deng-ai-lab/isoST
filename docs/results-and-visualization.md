# Results and Visualization

The tutorial notebooks write training artifacts, inference outputs, reconstructed PC volumes, and workflow-specific visualizations under local output directories.

## Training and Inference Outputs

Common outputs include:

- `experiments/model.pt`: trained model weights.
- `experiments/config.yml`: copied run configuration.
- `result/<index>_forward.npy`: inferred forward trajectories or intermediate slice outputs.
- `result/volume.npy`: reconstructed PC volume.
- `result/density.npy`: voxel density counts.
- `result/pc_volume.csv`: PC volume exported as a table.

Workflow-specific outputs include:

- Mouse brain: `result/log2_expr_220_all_pc.parquet`.
- Image registration: `result/log2_expr_264_all_pc.parquet`.
- Mouse embryo: `result/result.png`.
- Mouse brain and image registration notebooks can also save ROI predictions as `result/pred_roi.csv`.

## Visualization Inputs

Use these files as plotting inputs after post-processing:

- `result/volume.npy` for PC volume grids.
- `result/density.npy` for voxel density counts.
- `result/pc_volume.csv` for table-based PC inspection.
- `result/log2_expr_220_all_pc.parquet` or `result/log2_expr_264_all_pc.parquet` for recovered gene expression in the matching workflows.

The mouse brain and image registration workflows use Plotly-style 3D visualization patterns for point and voxel views. The embryo notebook uses Matplotlib to draw PC slices across z indices and saves `result/result.png`.

## Output Checklist

- Training finished and `experiments/model.pt` exists.
- Inference wrote the expected `result/<index>_forward.npy` files.
- Volume reconstruction wrote `result/volume.npy` and `result/density.npy`.
- `result/pc_volume.csv` exists and uses the expected PC count.
- Gene-expression parquet output exists for workflows that run PC to gene recovery.
- Visualization files are generated from the matching workflow outputs.

## Troubleshooting Checks

- If notebook imports fail, start Jupyter from the matching `script/<workflow>/` directory.
- If data files aren't found, confirm extracted data is under repository-root `data/` or update notebook paths.
- If the loader can't find slices, check the `shuffled_{slide_name}.pt` naming rule and the notebook `slide_names` list.
- If config and tensor shapes don't match, set `gene_dim` to the number of PC feature columns in the `.pt` files.
- If GPU memory is insufficient, use the provided training subset for the workflow before full inference.
