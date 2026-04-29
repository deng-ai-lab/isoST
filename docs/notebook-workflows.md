# Notebook Workflows

The tutorial notebooks show the intended end-to-end workflow: import project utilities, set paths, load a config, train isoST on a subset, run inference on the full slice set, then postprocess the result.

## Notebook Paths

- Mouse brain: `script/mouse_brain/run.ipynb`
- Image registration: `script/img_reg/run.ipynb`
- Mouse embryo: `script/mouse_embryo/run.ipynb`

Run each notebook from its own `script/<workflow>/` directory. Each notebook sets:

```python
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(project_root)
```

That means the working directory matters. Starting Jupyter from the repository root or another directory can make `project_root`, imports, config paths, and data paths resolve incorrectly.

## Common Steps

1. Import `biaxial_train` from `utils.train_ode` and `fine_inference` from `utils.inference`.
2. Fix random seeds for Python, NumPy, and PyTorch.
3. Define ordered `slide_names` for the workflow. The full lists live in the notebooks.
4. Set `proj`, `batch_num`, and `data_dir` for the training subset.
5. Load the workflow config from `config/mouse_brain.yml`, `config/img_reg.yml`, or `config/embryo.yml`.
6. Set training controls such as `device`, `checkpoint_every`, `backup_every`, `epochs`, and `mode`.
7. Create `experiments/` and `result/` directories for notebook outputs.
8. Train on the subset with `biaxial_train`.
9. Run full-data inference with `fine_inference`.
10. Postprocess inferred outputs into volumes, tables, and visualizations.

## Workflow Paths Used in Notebooks

Mouse brain uses:

- `proj = 'zhuang_ABCA_2/zscore_PC50_minmax'`
- Training data: `data/zhuang_ABCA_2/zscore_PC50_minmax/1_of_5_normPC_1`
- Full inference data: `data/zhuang_ABCA_2/zscore_PC50_minmax/1_of_1_normPC_1`

Image registration uses:

- `proj = 'zhuang_ABCA_3/zscore_PC50_minmax'`
- Training data: `data/zhuang_ABCA_3/zscore_PC50_minmax/1_of_16_normPC_1`
- Full inference data: `data/zhuang_ABCA_3/zscore_PC50_minmax/1_of_1_normPC_1`
- Image features: `data/CCFv3_feature`

Mouse embryo uses:

- `proj = 'mouse_embryo/combat_PC50_minmax'`
- Training data: `data/mouse_embryo/combat_PC50_minmax/1_of_100_normPC_1`
- Full inference data: `data/mouse_embryo/combat_PC50_minmax/1_of_1_normPC_1`

## Training Helper Arguments

The notebooks call `biaxial_train` with keyword arguments:

```python
biaxial_train(
    experiment_dir=experiment_dir,
    data_dir=data_dir,
    slide_names=slide_names,
    batch_num=1,
    config_file=config_file,
    device=device,
    checkpoint_every=checkpoint_every,
    backup_every=backup_every,
    epoch=epochs,
    mode=mode,
)
```

Argument meanings from the workflow context:

- `experiment_dir`: output directory for copied config, model weights, checkpoints, and logs.
- `data_dir`: preprocessed training subset directory.
- `slide_names`: ordered slice identifiers used by the loader.
- `batch_num`: batch or subset count passed to the trainer.
- `config_file`: YAML config for the workflow.
- `device`: PyTorch device string such as `cuda:0` or `cpu`.
- `checkpoint_every`: checkpoint interval in epochs.
- `backup_every`: backup checkpoint interval in epochs.
- `epoch`: list of epoch counts for the training phases.
- `mode`: training mode. The tutorials use `joint` to optimize shape and expression together.

## Inference Helper Arguments

The notebooks call `fine_inference` positionally:

```python
fine_inference(
    experiment_dir,
    total_data_dir,
    slide_names,
    mode,
    defined_d,
    result_dir,
    batch_num,
    device,
)
```

Argument meanings from the workflow context:

- `experiment_dir`: directory containing trained model outputs.
- `total_data_dir`: full preprocessed dataset for inference.
- `slide_names`: ordered slice identifiers for inference.
- `mode`: inference mode matching the trained model mode.
- `defined_d`: interpolation step, usually read from `config['params']['delta_d']`.
- `result_dir`: output directory for inferred trajectories and derived files.
- `batch_num`: workflow batch setting passed through to inference.
- `device`: PyTorch device string.

Do not replace these notebook calls with invented command-line training commands. The repository tutorials are notebook-based.
