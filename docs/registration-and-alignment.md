# Registration and alignment

The Zhuang MERFISH cells used by isoST come with CCFv3 coordinates published by the Allen Brain Cell Atlas. **isoST does not rerun the original image registration.** The upstream registration aligns tissue sections to the Allen Common Coordinate Framework (CCFv3); our preprocessing joins the published coordinates to MERFISH cells and samples an image feature volume in that space.

## Upstream registration and data

- The [Allen Zhuang MERFISH tutorial](https://alleninstitute.github.io/abc_atlas_access/notebooks/zhuang_merfish_tutorial.html#ccf-registration-and-parcellation-annotation) shows how to load each `Zhuang-ABCA-*-CCF/ccf_coordinates.csv` through `AbcProjectCache`, join it to cell metadata by `cell_label`, and obtain `x`, `y`, `z` and `parcellation_index`. These are **registered CCF coordinates**, distinct from the section-space `x`, `y`, `z` fields in `cell_metadata.csv`.
- The [original Zhuang atlas registration notebooks](https://github.com/ZhuangLab/whole_mouse_brain_MERFISH_atlas_scripts_2023/tree/main/scripts/ccf_registration) are published with the [whole-brain MERFISH atlas analysis code](https://github.com/ZhuangLab/whole_mouse_brain_MERFISH_atlas_scripts_2023). The folder contains specimen-specific notebooks such as [`registration_wb3_co1_final.ipynb`](https://github.com/ZhuangLab/whole_mouse_brain_MERFISH_atlas_scripts_2023/blob/main/scripts/ccf_registration/wb3_co1/registration_wb3_co1_final.ipynb), [`registration_wb3_co2_final.ipynb`](https://github.com/ZhuangLab/whole_mouse_brain_MERFISH_atlas_scripts_2023/blob/main/scripts/ccf_registration/wb3_co2/registration_wb3_co2_final.ipynb), and [`registration_wb3_sa1_final.ipynb`](https://github.com/ZhuangLab/whole_mouse_brain_MERFISH_atlas_scripts_2023/blob/main/scripts/ccf_registration/wb3_sa1/registration_wb3_sa1_final.ipynb). See [Zhang et al. (2023)](https://doi.org/10.1038/s41586-023-06808-9) for the study and registration methods.

The Allen tutorial notes that some peripheral sections could not be registered automatically and were manually oriented. A CCF coordinate file includes only cells for which registered coordinates were published; an inner join therefore reduces the cell set.

## Repository scripts and coordinate flow

1. [`script/prepare_zhuang_abca3_raw_subset.py`](../script/prepare_zhuang_abca3_raw_subset.py) selects Zhuang-ABCA-3 sections, joins official `ccf_coordinates.csv` to `cell_metadata.csv` by `cell_label`, and writes a subset H5AD. Its output keeps original section coordinates as `x_experiment`, `y_experiment`, `z_section` and registered CCF coordinates as `x`, `y`, `z`, with `parcellation_index`. It does not calculate a new registration transform.
2. [`script/extract_ccfv3_cell_features.py`](../script/extract_ccfv3_cell_features.py) maps those published CCF coordinates to the existing five-channel `data/CCFv3_feature/volume_downx2_features.pt` tensor and exports sampled features with `cell_label`, voxel indices and an `in_bounds` flag. Coordinates outside the volume retain `NaN` features.
3. During isoST-i training, [`model/image_regularized_model.py`](../model/image_regularized_model.py) loads that same volume and `physical_coordinates_v3.pt`; [`model/utils/DataExtraction.py`](../model/utils/DataExtraction.py) performs nearest-voxel feature lookup for predicted coordinates.

The volume is indexed `(z, y, x, feature)` while the Allen coordinate CSV is `(x, y, z)` in millimetres. The sampling script uses the model's 0.01 mm native spacing, downsampling factor 2, and nearest-voxel rounding: `voxel_zyx = round(ccf_xyz[::-1] / 0.02)`. Use coordinates and a feature volume from the same CCF reference and orientation. The script **samples the supplied feature tensor**; the method that originally constructed its five channels from atlas images is not present in this repository.

## Example

After downloading the official Zhuang-ABCA-3 files and extracting the isoST tutorial data:

```bash
python script/prepare_zhuang_abca3_raw_subset.py \
  --raw-h5ad /path/to/Zhuang-ABCA-3-raw.h5ad \
  --metadata /path/to/cell_metadata.csv \
  --coordinates /path/to/ccf_coordinates.csv \
  --output-h5ad /path/to/abca3_ccf_subset.h5ad

python script/extract_ccfv3_cell_features.py \
  --coordinates /path/to/ccf_coordinates.csv \
  --output /path/to/abca3_ccf_image_features.csv.gz
```

The second command can read a full official coordinate CSV in chunks. Its output preserves all rows; check `in_bounds` before using sampled values. The image registration tutorial notebook consumes the supplied preprocessed tensors and feature volume directly, so this CSV export is for inspecting or reusing the coordinate-to-image mapping rather than a required notebook input.
