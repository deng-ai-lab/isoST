# Configuration

The tutorial workflows load YAML files from the repository `config/` directory. The notebooks are the source of truth for which config is used by each workflow.

## Config Files

- Mouse brain uses `config/mouse_brain.yml`.
- Image registration uses `config/img_reg.yml`.
- Mouse embryo uses `config/embryo.yml`.

Each file has a top-level `trainer` name and a `params` block. The current trainers are `IsoST` for mouse brain and embryo, and `IsoSTImageReg` for image registration.

## Core Parameters

- `gene_dim`: number of expression features in each `.pt` tensor. The tutorials use 50 PCs, so this is `50`.
- `hidden_dim`: latent feature width used by the model.
- `head_num`: attention or graph head count used by the model implementation.
- `K`: model neighborhood or sampling parameter used by the trainer.
- `method`: integration method. The supplied configs use `euler`.
- `delta_d`: interpolation step along the z axis. Smaller values create finer interpolation and increase runtime.
- `stride`: loss computation interval along depth. The supplied tutorials use `1`.

## Optimization Parameters

- `lr`: learning rate.
- `optimizer_name`: optimizer selected by the trainer. The supplied configs use `NAdam`.
- `weight_decay`: weight decay passed to the optimizer.
- `warm_up_rate`: warm-up schedule setting used during training.

## Smoothing and Scheduling

- `std_x`, `std_y`, `std_z`: spatial smoothing or noise scale settings for each coordinate axis.
- `std_seq`: expression-feature smoothing or noise scale setting.
- `alpha`: weighting term used by the training objective.
- `dual`: enables bidirectional training and inference in configs that include it.
- `beta_start_value`, `beta_end_value`, `beta_start_iteration`, `beta_n_iterations`: beta schedule settings used during training.

## Image Registration Parameters

`config/img_reg.yml` includes fields that are specific to image-guided registration:

- `image_data_dir`: path to image-derived features. The config points to `data/CCFv3_feature`.
- `slice_data_dir`: path to the related slice data root. The config points to `data/zhuang_ABCA_3`.
- `slice_width`: width setting for slice handling.
- `spacing`: voxel spacing used by image registration logic.
- `scale_z`: z-axis scale setting.
- `template_sample_rate`: sampling rate for the image template.
- `_lambda_1`, `_lambda_2`: image-registration loss weights.

## Consistency Checks

- Match `gene_dim` to the tensor feature count after the first three coordinate columns.
- Keep `delta_d` aligned with the inference step `defined_d` used in the notebook.
- Use the config file for the matching notebook workflow.
- Treat notebook paths and ordered slide names as workflow-specific inputs rather than shared defaults.
