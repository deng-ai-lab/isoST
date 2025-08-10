import torch
import pandas as pd
import torch.nn as nn
from numpy import array
from .base_model import IsoST
from .utils.DataExtraction import extract_data_values
from .utils.losses import chamfer_distance
from .utils.modules_ODE import PCooStep, PUnetStep, PJointV2
import os
import yaml
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


class IsoSTImageReg(IsoST):
    def __init__(self, slice_data_dir, image_data_dir, scale_z=1, spacing=[0.01, 0.01, 0.01],
                 slice_width=0.4, template_sample_rate=0.125, _lambda_1=1, _lambda_2=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_width = torch.tensor(slice_width).to(self.device)
        self.spacing = torch.tensor(spacing).to(self.device)
        self.scale_z = scale_z
        self.w_img = _lambda_1 * self.w_coo
        self.w_exp = _lambda_2 * self.w_seq
        
        path_slice = os.path.join(PROJECT_ROOT, slice_data_dir)
        path_CCFv3 = os.path.join(PROJECT_ROOT, image_data_dir)

        self._load_geometry(path_slice)
        self._load_template(path_CCFv3, template_sample_rate)
        self._load_linear_model(path_slice)
        self.use_image_reg = True
        self.use_expr_reg = True

    def _load_geometry(self, slice_data_dir):
        min_x = pd.read_csv(f'{slice_data_dir}/min_dic.csv')['x'].values
        min_y = pd.read_csv(f'{slice_data_dir}/min_dic.csv')['y'].values
        min_z = pd.read_csv(f'{slice_data_dir}/min_dic.csv')['z'].values
        scale_xy = pd.read_csv(f'{slice_data_dir}/scale_dic.csv')['xy'].values
        self.origin = torch.tensor(array([min_x, min_y, min_z]), dtype=torch.float32).reshape(1, -1).to(self.device)
        self.scale = torch.tensor(array([scale_xy, scale_xy, [self.scale_z]]), dtype=torch.float32).reshape(1, -1).to(self.device)
        self.direction = torch.tensor([[0.0, 0.0, 1.0], [0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32).to(self.device)

    def _load_template(self, image_data_dir, template_sample_rate):
        self.template_image = torch.load(f'{image_data_dir}/volume_downx2_features.pt').to(self.device)
        self.img_feature_dim = self.template_image.shape[-1]
        self.min_value = torch.min(self.template_image).item()
        all_points = torch.load(f'{image_data_dir}/physical_coordinates_v3.pt').to(self.device)
        n = all_points.size(0)
        indices = torch.randperm(n)[:int(n * template_sample_rate)]
        self.template_points = (all_points[indices] - self.origin) / self.scale
        self.volume_downrate = 2

    def _load_linear_model(self, slice_data_dir):
        self.linear_frozen = LinearRegressionModel(self.gene_dim, self.img_feature_dim).to(self.device)
        weights = torch.load(f'{slice_data_dir}/linear_regression_weights.pth')
        self.linear_frozen.load_state_dict(weights)
        for p in self.linear_frozen.parameters():
            p.requires_grad = False
    
    def _init_modules(self):
        self.f_coo = PCooStep(3, self.hidden_dim, 3, self.std_x, self.std_y, self.std_z, self.sqrt_dd)
        self.g_coo = PCooStep(3, self.hidden_dim, 3, self.std_x, self.std_y, self.std_z, self.sqrt_dd)
        self.f_seq = PUnetStep(in_channels=3 + self.gene_dim, hidden_channels=self.hidden_dim, depth=3,
                               out_channels=self.gene_dim, std=self.std_seq, sqrt_d=self.sqrt_dd)
        self.g_seq = PUnetStep(in_channels=3 + self.gene_dim, hidden_channels=self.hidden_dim, depth=3,
                               out_channels=self.gene_dim, std=self.std_seq, sqrt_d=self.sqrt_dd)
        self.f_joint = PJointV2(self.f_coo, self.f_seq, topk=5)
        self.g_joint = PJointV2(self.g_coo, self.g_seq, topk=5)

    def _compute_image_loss(self, pred, template, depth):
        z = self.template_points[:, 2]
        mask = (z >= depth - self.slice_width / 2) & (z < depth + self.slice_width / 2)
        return chamfer_distance(pred[:, :2], self.template_points[mask][:, :2])

    def _compute_expression_loss(self, pred):
        coo_pred = pred[:, :3]
        exp_pred = pred[:, 3:]
        img = extract_data_values(self.template_image, self.volume_downrate, coo_pred,
                                  self.direction, self.spacing, self.origin, self.scale, self.min_value)
        pred_img = self.linear_frozen(exp_pred)
        return torch.mean((pred_img - img) ** 2)

    def _compute_total_loss(self, u_lists, depth_dir, model, loss_type, reversed=False):
        loss = super()._compute_total_loss(u_lists, depth_dir, model, loss_type, reversed)
        if loss_type in ['coo', 'joint']:
            for kk in range(len(u_lists) - self.stride):
                u0 = u_lists[kk]
                depth0 = torch.mean(u0[:, 2])
                trajectory = self._compute_feature_trajectory(model, u0, depth_dir[kk])
                for i in range(1, len(depth_dir[kk]) - 1):
                    loss += self.w_img * self._compute_image_loss(trajectory[i], self.template_points, depth_dir[kk][i])
        if loss_type in ['seq']:
            for kk in range(len(u_lists) - self.stride):
                u0 = u_lists[kk]
                trajectory = self._compute_feature_trajectory(model, u0, depth_dir[kk])
                for i in range(1, len(depth_dir[kk]) - 1):
                    loss += self.w_exp * self._compute_expression_loss(trajectory[i])
        return loss
