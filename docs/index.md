# isoST Documentation

This chapter set keeps the detailed tutorial material outside the concise project README. Start with the notebook that matches your workflow, then use the data, configuration, and post-processing chapters as references while reading the notebook cells.

## Chapters

- [Data format and layout](data.md): input `.pt` tensors, naming rules, normalization metadata, PCA files, and workflow data paths.
- [Configuration](configuration.md): root YAML files and the meaning of key model and training parameters.
- [Notebook workflows](notebook-workflows.md): how the tutorial notebooks are organized, where to run them from, and the training and inference helper arguments.
- [Post-processing](post-processing.md): volume reconstruction, density outputs, PC tables, and PC to gene expression recovery.
- [Results and visualization](results-and-visualization.md): common output files, workflow-specific products, plotting inputs, and final checks.

## Tutorial Entry Points

The repository includes three tutorial notebooks:

- `script/mouse_brain/run.ipynb`
- `script/img_reg/run.ipynb`
- `script/mouse_embryo/run.ipynb`

Run each notebook from its own `script/<workflow>/` directory. The notebooks compute `project_root` with `../../`, so running them from another directory can point imports, configs, and data paths at the wrong location.

## Data Setup Reminder

The downloader is `script/download_data.py`. For the least notebook path editing, extract the archive into the repository root so the expected `data/` directory is populated.
