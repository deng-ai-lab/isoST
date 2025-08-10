import os
import re
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

class VolumeProcessor:
    def __init__(self, data_dir, result_dir, volume_size, gene_list, max_lence):
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.volume_size = volume_size  # (x_size, y_size, z_size)
        self.scale_info = pd.read_csv(os.path.join(self.data_dir, "scale_dic.csv"))
        self.min_info = pd.read_csv(os.path.join(self.data_dir, "min_dic.csv"))

        self.z_lence = max_lence
        self.cube_size = 1 / self.z_lence
        self.gene_list = gene_list

    def get_data_coordinate_info(self):
        print("Starting load_result...")
        pattern = re.compile(r"(\d+)_(\w+)\.npy")
        infer_list = []
        for file_name in os.listdir(self.result_dir):
            match = pattern.match(file_name)
            if match:
                number, source = match.groups()
                if source == 'forward':
                    infer_list.append(number)

        infer_list = sorted(infer_list, key=lambda x: int(x))
        all_number = len(infer_list)
        team_x = []
        team_y = []

        for number in tqdm(range(all_number), desc="Loading inferred results"):
            data = np.load(os.path.join(self.result_dir, f'{number}_forward.npy'))
            team_x += (data[:, 0]).tolist()
            team_y += (data[:, 1]).tolist()

        min_dic_x = np.min(team_x)
        min_dic_y = np.min(team_y)

        scale_x = np.max(team_x) - np.min(team_x)
        scale_y = np.max(team_y) - np.min(team_y)
        scale_dic_xy = np.max([scale_x, scale_y])
        return min_dic_x, min_dic_y, scale_dic_xy


    def result_to_time_volume(self, n_features, z_scale):
        print("Starting load_result...")

        pattern = re.compile(r"(\d+)_(\w+)\.npy")
        infer_list = []

        for file_name in os.listdir(self.result_dir):
            match = pattern.match(file_name)
            if match:
                number, source = match.groups()
                if source == 'forward':
                    infer_list.append(number)

        infer_list = sorted(infer_list, key=lambda x: int(x))
        all_number = len(infer_list)

        min_dic_x, min_dic_y, scale_dic_xy = self.get_data_coordinate_info()
        data_numpy_cache = {}
        for number in tqdm(range(all_number), desc="Loading inferred results"):
            data = np.load(os.path.join(self.result_dir, f'{number}_forward.npy'))
            data[:, 0] = (data[:, 0] - min_dic_x) / scale_dic_xy
            data[:, 1] = (data[:, 1] - min_dic_y) / scale_dic_xy
            data[:, 2] = number
            for j in range(data.shape[-1] - 3):
                scale = self.scale_info[f'PC_{j+1}'].values
                offset = self.min_info[f'PC_{j+1}'].values
                data[:, 3 + j] = data[:, 3 + j] * scale + offset
            data_numpy_cache[f'{number}_forward'] = data

        print("Finished load_result.")


        print("Starting scatter_to_volume...")
        volume_x = math.ceil(self.volume_size[0] / self.cube_size)
        volume_y = math.ceil(self.volume_size[1] / self.cube_size)
        volume_z = math.ceil(self.volume_size[2])

        volume = np.full((volume_x, volume_y, volume_z, n_features), np.nan)
        count = np.zeros((volume_x, volume_y, volume_z))
        cube_size_array = np.array([self.cube_size, self.cube_size, 1])
        slice_set = data_numpy_cache
        for key in tqdm(slice_set, desc="Processing slices"):
            points, features = slice_set[key][:, :3], slice_set[key][:, 3:]
            indices = np.floor(points / cube_size_array).astype(int)
            for i, (x, y, z) in enumerate(indices):
                if 0 <= x < volume_x and 0 <= y < volume_y and 0 <= z < volume_z:
                    if np.isnan(volume[x, y, z, 0]).any():
                        volume[x, y, z] = features[i]
                    else:
                        volume[x, y, z] += features[i]
                    count[x, y, z] += 1

        valid = count > 0
        volume[valid] /= count[valid][:, np.newaxis]
        print("Finished scatter_to_volume.")
        return volume, count


    def result_to_volume(self, n_features, z_scale=None, swamp=False):
        print("Starting load_result...")

        pattern = re.compile(r"(\d+)_(\w+)\.npy")
        infer_list = []

        for file_name in os.listdir(self.result_dir):
            match = pattern.match(file_name)
            if match:
                number, source = match.groups()
                if source == 'forward':
                    infer_list.append(number)

        infer_list = sorted(infer_list, key=lambda x: int(x))
        all_number = len(infer_list)

        data_numpy_cache = {}
        for number in tqdm(range(all_number), desc="Loading inferred results"):
            data = np.load(os.path.join(self.result_dir, f'{number}_forward.npy'))
            if z_scale is None:
                data[:, 2] /= self.scale_info['xy'].values
            else:
                data[:, 2] /= z_scale
            if swamp:
                data[:, [0, 2]] = data[:, [2, 0]]  # swap x and z
            for j in range(data.shape[-1] - 3):
                scale = self.scale_info[f'PC_{j+1}'].values
                offset = self.min_info[f'PC_{j+1}'].values
                data[:, 3 + j] = data[:, 3 + j] * scale + offset
            data_numpy_cache[f'{number}_forward'] = data

        print("Finished load_result.")


        print("Starting scatter_to_volume...")
        volume_x = math.ceil(self.volume_size[0] / self.cube_size)
        volume_y = math.ceil(self.volume_size[1] / self.cube_size)
        volume_z = math.ceil(self.volume_size[2] / self.cube_size)

        volume = np.full((volume_x, volume_y, volume_z, n_features), np.nan)
        count = np.zeros((volume_x, volume_y, volume_z))

        slice_set = data_numpy_cache
        for key in tqdm(slice_set, desc="Processing slices"):
            points, features = slice_set[key][:, :3], slice_set[key][:, 3:]
            indices = np.floor(points / self.cube_size).astype(int)
            for i, (x, y, z) in enumerate(indices):
                if 0 <= x < volume_x and 0 <= y < volume_y and 0 <= z < volume_z:
                    if np.isnan(volume[x, y, z, 0]).any():
                        volume[x, y, z] = features[i]
                    else:
                        volume[x, y, z] += features[i]
                    count[x, y, z] += 1

        valid = count > 0
        volume[valid] /= count[valid][:, np.newaxis]
        print("Finished scatter_to_volume.")
        return volume, count

    def volume_to_df(self, data, genelist=None):
        print("Starting volume_to_df...")
        non_nan_indices = ~np.isnan(data).any(axis=-1)
        filtered_data = data[non_nan_indices]
        x_indices, y_indices, z_indices = np.nonzero(non_nan_indices)
        flattened_data = filtered_data.reshape(-1, data.shape[-1])
        index_data = np.column_stack((x_indices, y_indices, z_indices))
        columns = ['x', 'y', 'z'] + (genelist if genelist else [f'PC{i+1}' for i in range(data.shape[-1])])
        df = pd.DataFrame(np.hstack((index_data, flattened_data)), columns=columns)
        print("Finished volume_to_df.")
        return df

    def process_chunk(self, df_chunk):
        print("Starting process_chunk...")
        volume_x, volume_y, volume_z = self.volume_size
        volume = np.full((volume_x, volume_y, volume_z, len(self.gene_list)), np.nan)

        points = df_chunk.values[:, :3].astype(int)
        features = df_chunk[self.gene_list].values

        mask = (points[:, 0] >= 0) & (points[:, 0] < volume_x) & \
               (points[:, 1] >= 0) & (points[:, 1] < volume_y) & \
               (points[:, 2] >= 0) & (points[:, 2] < volume_z)

        valid_indices = points[mask]
        valid_features = features[mask]

        volume[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = valid_features
        print("Finished process_chunk.")
        return volume

    def df_to_volume(self, df):
        print("Starting df_to_volume...")
        volume_x, volume_y, volume_z = self.volume_size
        volume = np.full((volume_x, volume_y, volume_z, len(self.gene_list)), np.nan)

        points = df[['x', 'y', 'z']].values.astype(int)
        features = df[self.gene_list].values

        volume[points[:, 0], points[:, 1], points[:, 2]] = features
        print("Finished df_to_volume.")
        return volume

    def pc_to_expression(self, volume, pc_model, z_length):
        print("Starting pc_to_expression...")
        pcs = self.volume_to_df(volume, genelist=[f'PC{i+1}' for i in range(volume.shape[-1])])
        X_reconstructed = pc_model.inverse_transform(pcs[[f'PC{i+1}' for i in range(50)]])
        log_expr = pd.DataFrame(X_reconstructed)

        predictions = pcs[['x', 'y', 'z']].copy()
        predictions = pd.concat([predictions, log_expr], axis=1)
        predictions.columns = ['x', 'y', 'z'] + self.gene_list
        predictions = predictions.astype('float32')

        os.makedirs(self.result_dir, exist_ok=True)
        output_path = f"{self.result_dir}/log2_expr_{z_length}_all_pc.parquet"
        pq.write_table(pa.Table.from_pandas(predictions), output_path)
        print("Finished pc_to_expression.")
        return output_path