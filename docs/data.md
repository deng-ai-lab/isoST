# Data Format and Layout

isoST tutorials use preprocessed serial slice tensors plus metadata for normalization and PC to gene recovery. The supplied dataset layouts differ, so use the workflow-specific paths in this chapter instead of assuming one shared directory shape.

## Slice Tensor Format

Each input slice is stored as a PyTorch `.pt` tensor with shape `N x (3 + feature_dim)`:

- Columns 1 to 3 store spatial coordinates `(x, y, z)`, where `z` is the interpolation axis.
- Remaining columns store expression features. The supplied tutorials use 50 principal components.
- The loader expects files named `shuffled_{slide_name}.pt`.

The `slide_names` list in each notebook must match the dataset files after the loader adds the `shuffled_` prefix and `.pt` suffix. Keep the slice order consistent along the z axis. The notebooks contain the full ordered slide-name lists, so this documentation summarizes the rule rather than repeating those long lists.

## Normalization

The supplied tensors are already preprocessed for the tutorial workflows:

- `x` and `y` coordinates are shifted by each axis minimum.
- Both spatial axes are scaled by the larger width across `x` and `y`, which keeps xy scaling isotropic within the slice.
- PC features are min-max normalized per feature.

The normalization metadata is used later during post-processing and recovery:

- `min_dic.csv` stores minimum values for spatial and PC dimensions.
- `scale_dic.csv` stores scaling factors for spatial and PC dimensions.

Keep these metadata files with the corresponding PC data directory. Missing or mismatched metadata can break de-normalization and downstream volume export.

## Gene and PCA Files

The gene list and PCA model define the feature space used for PC recovery:

- `gene.csv` stores the gene symbols used in the PCA feature space. Mouse brain and image registration notebooks read `gene_symbol`; the embryo notebook reads `gene`.
- `zscore_pc_model.pkl` is loaded with joblib for inverse PCA transformation in the zscore workflows.

Use the PCA model that belongs to the same workflow and PC count as the tensors. A model trained for a different feature space won't recover the intended gene expression values.

## Workflow Data Paths

The downloader is `script/download_data.py`. After extraction into the repository root, the notebooks expect these paths under `data/`.

### Mouse Brain

- Notebook: `script/mouse_brain/run.ipynb`
- Config: `config/mouse_brain.yml`
- Training subset: `data/zhuang_ABCA_2/zscore_PC50_minmax/1_of_5_normPC_1`
- Full inference data: `data/zhuang_ABCA_2/zscore_PC50_minmax/1_of_1_normPC_1`
- Gene list: `data/zhuang_ABCA_2/gene.csv`
- PCA model: `data/zhuang_ABCA_2/zscore_PC50_minmax/zscore_pc_model.pkl`

### Image Registration

- Notebook: `script/img_reg/run.ipynb`
- Config: `config/img_reg.yml`
- Training subset: `data/zhuang_ABCA_3/zscore_PC50_minmax/1_of_16_normPC_1`
- Full inference data: `data/zhuang_ABCA_3/zscore_PC50_minmax/1_of_1_normPC_1`
- CCFv3 features: `data/CCFv3_feature`
- Gene list: `data/zhuang_ABCA_3/gene.csv`
- PCA model: `data/zhuang_ABCA_3/zscore_PC50_minmax/zscore_pc_model.pkl`

### Mouse Embryo

- Notebook: `script/mouse_embryo/run.ipynb`
- Config: `config/embryo.yml`
- Training subset: `data/mouse_embryo/combat_PC50_minmax/1_of_100_normPC_1`
- Full inference data: `data/mouse_embryo/combat_PC50_minmax/1_of_1_normPC_1`
- Gene list: `data/mouse_embryo/gene.csv`

## Data Checks

- Confirm the `.pt` tensor feature count matches `gene_dim` in the workflow config.
- Confirm files follow the `shuffled_{slide_name}.pt` naming rule.
- Confirm `min_dic.csv` and `scale_dic.csv` are present for workflows that recover scaled values.
- Confirm the PCA model matches the PC count before running PC to gene recovery.
