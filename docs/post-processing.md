# Post-Processing

Post-processing converts inferred PC trajectories into dense volumes, exports tabular PC values, and recovers gene expression where the workflow includes a matching PCA model.

## VolumeProcessor Setup

The notebooks use `VolumeProcessor` from `utils.postprocess` after inference has written outputs into `result/`.

Mouse brain and image registration read the gene list from `gene_symbol`:

```python
gene = pd.read_csv(gene_path, index_col=0)
processor = VolumeProcessor(
    data_dir=data_dir,
    result_dir=result_dir,
    volume_size=(1.0, 0.8, 0.5),
    gene_list=gene['gene_symbol'].tolist(),
    max_lence=220,
)
```

The embryo notebook reads the gene list from `gene` and uses a time-volume shape:

```python
gene = pd.read_csv(gene_path, index_col=0)
processor = VolumeProcessor(
    data_dir=data_dir,
    result_dir=result_dir,
    volume_size=(1.0, 1.0, 61),
    gene_list=gene['gene'].tolist(),
    max_lence=256,
)
```

Use the values from the matching notebook when reproducing a workflow. They are not universal across datasets.

## Reconstructing a PC Volume

Mouse brain and image registration use `result_to_volume` in the tutorial pattern:

```python
volume, count = processor.result_to_volume(n_features=50, swamp=True)
pc_df = processor.volume_to_df(volume)

np.save(f'{result_dir}/volume.npy', volume)
np.save(f'{result_dir}/density.npy', count)
pc_df.to_csv(f'{result_dir}/pc_volume.csv')
```

Mouse embryo uses `result_to_time_volume`:

```python
volume, count = processor.result_to_time_volume(n_features=50, z_scale=0.1)
pc_df = processor.volume_to_df(volume)

np.save(f'{result_dir}/volume.npy', volume)
np.save(f'{result_dir}/density.npy', count)
pc_df.to_csv(f'{result_dir}/pc_volume.csv')
```

The common outputs are:

- `result/volume.npy`: reconstructed PC volume.
- `result/density.npy`: voxel density counts.
- `result/pc_volume.csv`: PC volume exported as a table.

## PC to Gene Recovery

For zscore PC workflows, the notebooks load the matching PCA model with joblib and call `pc_to_expression`:

```python
import joblib

pc_model = joblib.load(model_path)
processor.pc_to_expression(volume, pc_model, 220)
```

The mouse brain workflow writes `result/log2_expr_220_all_pc.parquet`. The image registration workflow writes `result/log2_expr_264_all_pc.parquet`. Use the output length and model path from the matching notebook.

The embryo notebook shown in this repository focuses on PC volume reconstruction and slice visualization. It doesn't run the same PCA inverse-transform step in the visible workflow.

## Recovery Checks

- Use the PCA model from the same data root as the PC tensors.
- Match the PC count passed as `n_features` to `gene_dim` in the config.
- Check that `gene.csv` has the expected column name for the workflow.
- Confirm `volume.npy`, `density.npy`, and `pc_volume.csv` exist before trying gene recovery.
