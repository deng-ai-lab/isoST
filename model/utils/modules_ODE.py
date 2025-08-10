import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.graph_unet import GraphUNet
from torch_cluster import knn_graph
from .DataExtraction import extract_data_values
import torch.nn.functional as F

dropout = 0

class GNN(nn.Module):
    def __init__(self, input_dim, z_dim, output_dim, head, topk=5, dropout=dropout):
        super(GNN, self).__init__()
        dim_list = [input_dim, z_dim, output_dim]
        # Graph transformer layers
        self.tfc_1 = TransformerConv(in_channels=dim_list[0], out_channels=dim_list[1], heads=head, dropout=dropout)
        self.tfc_2 = TransformerConv(in_channels=dim_list[1] * head, out_channels=dim_list[1] // head, heads=head,
                                     dropout=dropout)
        self.tfc_3 = TransformerConv(in_channels=dim_list[1] * head, out_channels=dim_list[-1] // head, heads=head,
                                     dropout=dropout)
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU(inplace=True)
        self.topk = topk


class p_coo(GNN):
    def __init__(self, input_dim, z_dim, output_dim=3, std=0.01, head=1, topk=5, dropout=dropout):
        super(p_coo, self).__init__(input_dim, z_dim, output_dim=output_dim, head=head, topk=topk, dropout=dropout)

    def forward(self, t, x):
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        x = self.tfc_1(x, edge_index)
        x = self.tanh(x)
        x = self.tfc_2(x, edge_index) + x
        x = self.tanh(x)
        drift_coo = self.tfc_3(x, edge_index)
        return drift_coo


class p_seq(GNN):
    def __init__(self, input_dim, z_dim, output_dim, coo_net, topk=5, std=0.01, head=1, dropout=dropout):
        super(p_seq, self).__init__(input_dim, z_dim, output_dim, head, topk=topk, dropout=dropout)
        self.coo_net = coo_net

    def forward(self, t, x):
        coo = x[:, :3]
        drift_coo = self.coo_net(t, coo)
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        x = self.tfc_1(x, edge_index)
        x = self.silu(x)
        x = self.tfc_2(x, edge_index) + x
        x = self.silu(x)
        drift_seq = self.tfc_3(x, edge_index)
        drift_ = torch.cat([drift_coo, drift_seq], dim=1)
        return drift_


class p_coo_norm(GNN):
    def __init__(self, input_dim, z_dim, output_dim=3, std=0.01, head=1, topk=5, dropout=dropout):
        super(p_coo_norm, self).__init__(input_dim, z_dim, output_dim=output_dim, head=head, topk=topk, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.tfc_1(x, edge_index)
        x = self.tanh(x)

        x = self.tfc_2(x, edge_index) + x
        x = self.tanh(x)

        drift_coo = self.tfc_3(x, edge_index)
        return drift_coo



# class p_coo_step(GNN):
#     def __init__(self, input_dim, z_dim, output_dim=3, std=0.01, step, head=1, topk=5, dropout=dropout):
#         super(p_coo_norm, self).__init__(input_dim, z_dim, output_dim=output_dim, head=head, topk=topk, dropout=dropout)
#
#     def forward(self, x, edge_index):
#         x = self.tfc_1(x, edge_index)
#         x = self.tanh(x)
#
#         x = self.tfc_2(x, edge_index) + x
#         x = self.tanh(x)
#
#         drift_coo = self.tfc_3(x, edge_index)
#         return drift_coo


class p_seq_norm(GNN):
    def __init__(self, input_dim, z_dim, output_dim, topk=5, std=0.01, head=1, dropout=dropout):
        super(p_seq_norm, self).__init__(input_dim, z_dim, output_dim, head, topk=topk, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(z_dim)
        self.bn3 = nn.BatchNorm1d(z_dim)

    def forward(self, x_seq, edge_index):
        x = self.bn1(x_seq)
        x = self.tfc_1(x, edge_index)
        x = self.silu(x)

        x = self.bn2(x)
        x = self.tfc_2(x, edge_index) + x
        x = self.silu(x)

        x = self.bn3(x)
        drift_seq = self.tfc_3(x, edge_index)
        return drift_seq


class PUnet(nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 depth=2,
                 pool_ratios=0.5,
                 sum_res=True,
                 act=torch.tanh):
        super(PUnet, self).__init__()
        self.net = GraphUNet(in_channels, hidden_channels, out_channels, depth, pool_ratios, sum_res, act)

    def forward(self, x, edge_index):
        drift_coo = self.net(x, edge_index)
        return drift_coo

class PCooStep(GNN):
    def __init__(self, input_dim, z_dim, output_dim=3,
                 std_x=0.01,
                 std_y=0.01,
                 std_z=0.01,
                 sqrt_d=0.1,
                 z_distribute='Gaussian', head=1, topk=5, dropout=dropout):
        super(PCooStep, self).__init__(input_dim, z_dim, output_dim=output_dim, head=head, topk=topk, dropout=dropout)
        self.sigma = [std_x, std_y, std_z]
        self.sqrt_d = sqrt_d
        self.z_distribute = z_distribute

    def forward(self, x, edge_index):
        x = self.tfc_1(x, edge_index)
        x = self.tanh(x)

        x = self.tfc_2(x, edge_index) + x
        x = self.tanh(x)

        if self.sigma[2] > 1e-4:
            drift_coo = self.tfc_3(x, edge_index)
            z = torch.randn_like(drift_coo, device=x.device)
            z[:, 0] *= self.sigma[0]
            z[:, 1] *= self.sigma[1]
            z[:, 2] *= self.sigma[2]
        else:
            drift_coo_ = self.tfc_3(x, edge_index)
            ones_column = torch.ones(drift_coo_.shape[0], 1, device=drift_coo_.device)
            drift_coo = torch.cat([drift_coo_, ones_column], dim=1)
            z = torch.randn_like(drift_coo, device=x.device)
            z[:, 0] *= self.sigma[0]
            z[:, 1] *= self.sigma[1]
            z[:, 2] *= 0
        return drift_coo + z / self.sqrt_d


class PUnetStep(nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 std=0.01,
                 sqrt_d=0.1,
                 depth=2,
                 pool_ratios=0.5,
                 sum_res=True,
                 act=torch.tanh):
        super(PUnetStep, self).__init__()
        self.net = GraphUNet(in_channels, hidden_channels, out_channels, depth, pool_ratios, sum_res, act)
        self.std = std
        self.sqrt_d = sqrt_d

    def forward(self, x, edge_index):
        drift_coo = self.net(x, edge_index)
        z = torch.randn_like(drift_coo, device=x.device)
        return drift_coo + self.std * z / self.sqrt_d


class PUnetStepLayered(nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 std=0.01,
                 sqrt_d=0.1,
                 depth=2,
                 pool_ratios=0.5,
                 sum_res=True,
                 act=torch.tanh):
        super(PUnetStepLayered, self).__init__()
        self.net = GraphUNet(in_channels, hidden_channels, out_channels, depth, pool_ratios, sum_res, act)
        self.std = std
        self.sqrt_d = sqrt_d

    def forward(self, x, edge_index):
        drift_coo = self.net(x, edge_index)
        z = torch.randn_like(drift_coo, device=x.device)
        drift_coo_noised = drift_coo + self.std * z / self.sqrt_d


        drift_label = torch.zeros_like(x[:, 3:], device=x.device)
        return drift_coo_noised, drift_label


class PUnetStepV2(nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 std=0.01,
                 sqrt_d=0.1,
                 depth=2,
                 pool_ratios=0.5,
                 sum_res=True,
                 act=torch.tanh):
        super(PUnetStepV2, self).__init__()
        self.net = GraphUNet(in_channels, hidden_channels, out_channels, depth, pool_ratios, sum_res, act)
        self.std = std
        self.sqrt_d = sqrt_d

    def forward(self, x, edge_index):
        drift_coo = self.net(x, edge_index)
        if self.training:  # Only add noise during training
            z = torch.randn_like(drift_coo, device=x.device)
            return drift_coo + self.std * z / self.sqrt_d
        else:
            # print('No noise added during evaluation')
            return drift_coo


class PseqStepV3(GNN):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 std=0.01,
                 sqrt_d=0.1,
                 head=1, topk=5, dropout=dropout):
        super(PseqStepV3, self).__init__(in_channels, hidden_channels, out_channels,
                                         head=head, topk=topk, dropout=dropout)
        self.std = std
        self.sqrt_d = sqrt_d

    def forward(self, x, edge_index):
        x = self.tfc_1(x, edge_index)
        x = self.silu(x)

        x = self.tfc_2(x, edge_index) + x
        x = self.silu(x)

        drift_seq = self.tfc_3(x, edge_index)
        # if self.training:  # Only add noise during training
        #     z = torch.randn_like(drift_seq, device=x.device)
        #     return drift_seq + self.std * z / self.sqrt_d
        # else:
        #     # print('No noise added during evaluation')
        #     return drift_seq
        # return drift_seq
        z = torch.randn_like(drift_seq, device=x.device)
        return drift_seq + self.std * z / self.sqrt_d


class p_joint(nn.Module):
    def __init__(self, p_coo, p_seq, topk):
        super(p_joint, self).__init__()
        self.p_coo = p_coo
        self.p_seq = p_seq
        self.topk = topk

    def forward(self, t, x):
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        coo_out = self.p_coo(x[:, :3], edge_index)
        seq_out = self.p_seq(x[:, 3:], edge_index)
        drift_ = torch.cat([coo_out, seq_out], dim=1)
        return drift_


class PJointV2(nn.Module):
    def __init__(self, p_coo, p_seq, topk):
        super(PJointV2, self).__init__()
        self.p_coo = p_coo
        self.p_seq = p_seq
        self.topk = topk

    def forward(self, t, x):
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        coo_out = self.p_coo(x[:, :3], edge_index)
        seq_out = self.p_seq(x, edge_index)
        drift_ = torch.cat([coo_out, seq_out], dim=1)
        return drift_
    

class PJointLabel(nn.Module):
    def __init__(self, p_coo, p_seq, topk):
        super(PJointLabel, self).__init__()
        self.p_coo = p_coo
        self.p_seq = p_seq
        self.topk = topk

    def forward(self, t, x):
        mu_coo = x[:, :3]
        spatial_coo = mu_coo
        batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
        edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        coo_out, seq_out = self.p_seq(x, edge_index)
        drift_ = torch.cat([coo_out, seq_out], dim=1)
        return drift_


class PJointV3(nn.Module):
    def __init__(self, p_coo, p_seq, topk):
        super(PJointV3, self).__init__()
        self.p_coo = p_coo
        self.p_seq = p_seq
        self.topk = topk

    def forward(self, t, x):
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        coo_out = self.p_coo(x[:, :3], edge_index)
        seq_out = self.p_seq(x, edge_index)
        drift_ = torch.cat([coo_out, seq_out], dim=1)
        return drift_


class PJointImg(nn.Module):
    def __init__(self, p_coo, p_seq, volume, direction, spacing, origin, scale, topk):
        super(PJointImg, self).__init__()
        self.p_coo = p_coo
        self.p_seq = p_seq

        self.volume = volume
        self.direction = direction
        self.spacing = spacing
        self.origin = origin
        self.scale = scale

        self.min_value = torch.min(volume).item()

        self.topk = topk

    def forward(self, t, x):
        image = extract_data_values(data=self.volume,
                                    coords=x[:, :3],
                                    direction=self.direction,
                                    spacing=self.spacing,
                                    origin=self.origin,
                                    scale=self.scale,
                                    padding_value=self.min_value)
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        coo_out = self.p_coo(torch.cat([x[:, :3], image], dim=1), edge_index)
        seq_out = self.p_seq(torch.cat([x, image], dim=1), edge_index)
        drift_ = torch.cat([coo_out, seq_out], dim=1)
        return drift_


# class PJointImgV2(nn.Module):
#     def __init__(self, p_coo, p_seq, volume, direction, spacing, origin, scale, topk):
#         super(PJointImgV2, self).__init__()
#         self.p_coo = p_coo
#         self.p_seq = p_seq

#         self.volume = volume
#         self.direction = direction
#         self.spacing = spacing
#         self.origin = origin
#         self.scale = scale

#         self.min_value = torch.min(volume).item()

#         self.topk = topk

#     def forward(self, t, x):
#         image = extract_data_values(data=self.volume,
#                                     coords=x[:, :3],
#                                     direction=self.direction,
#                                     spacing=self.spacing,
#                                     origin=self.origin,
#                                     scale=self.scale,
#                                     padding_value=self.min_value)
#         with torch.no_grad():
#             mu_coo = x[:, :3]
#             spatial_coo = mu_coo
#             batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
#             edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
#         coo_out = self.p_coo(x[:, :3], edge_index)
#         seq_out = self.p_seq(torch.cat([x, image], dim=1), edge_index)
#         drift_ = torch.cat([coo_out, seq_out], dim=1)
#         return drift_

class PJointImgV2(nn.Module):
    def __init__(self, p_coo, p_seq,
                  volume, min_value, volume_downrate, 
                  direction, spacing, origin, scale, topk):
        super(PJointImgV2, self).__init__()
        self.p_coo = p_coo
        self.p_seq = p_seq

        self.volume = volume
        self.volume_downrate = volume_downrate
        self.direction = direction
        self.spacing = spacing
        self.origin = origin
        self.scale = scale

        self.min_value = min_value
        self.topk = topk

    def forward(self, t, x):
        image = extract_data_values(data=self.volume, 
                                    downrate=self.volume_downrate,
                                    coords=x[:, :3],
                                    direction=self.direction,
                                    spacing=self.spacing,
                                    origin=self.origin,
                                    scale=self.scale,
                                    padding_value=self.min_value)
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        coo_out = self.p_coo(x[:, :3], edge_index)
        seq_out = self.p_seq(torch.cat([x, image], dim=1), edge_index)
        drift_ = torch.cat([coo_out, seq_out], dim=1)
        return drift_

class PJointImgV3(nn.Module):
    def __init__(self, p_coo, p_seq, volume, volume_downrate, direction, spacing, origin, scale, topk):
        super(PJointImgV2, self).__init__()
        self.p_coo = p_coo
        self.p_seq = p_seq

        self.volume = volume
        self.volume_downrate = volume_downrate
        self.direction = direction
        self.spacing = spacing
        self.origin = origin
        self.scale = scale

        self.min_value = torch.min(volume).item()

        self.topk = topk

    def forward(self, t, x):
        image = extract_data_values(data=self.volume, 
                                    downrate=self.volume_downrate,
                                    coords=x[:, :3],
                                    direction=self.direction,
                                    spacing=self.spacing,
                                    origin=self.origin,
                                    scale=self.scale,
                                    padding_value=self.min_value)
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        coo_out = self.p_coo(x[:, :3], edge_index)
        seq_out = self.p_seq(torch.cat([x, image], dim=1), edge_index)
        drift_ = torch.cat([coo_out, seq_out], dim=1)
        return drift_


class p_coo_noise(GNN):
    def __init__(self, input_dim, z_dim, output_dim=3, std=0.01, sqrt_d=0.1, head=1, topk=5, dropout=dropout):
        super(p_coo_noise, self).__init__(input_dim, z_dim, output_dim=output_dim, head=head, topk=topk,
                                          dropout=dropout)
        self.noise_std = std
        self.sqrt_d = sqrt_d

    def forward(self, t, x):
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        x = self.tfc_1(x, edge_index)
        x = self.tanh(x)
        x = self.tfc_2(x, edge_index) + x
        x = self.tanh(x)
        drift_coo = self.tfc_3(x, edge_index)

        # if self.training:
        #     noise = torch.randn_like(drift_coo) * self.noise_std
        #     drift_coo += noise
        noise = torch.randn_like(drift_coo) * self.noise_std * self.sqrt_d
        drift_coo += noise

        return drift_coo


class p_seq_noise(GNN):
    def __init__(self, input_dim, z_dim, output_dim, coo_net, topk=5, std=0.01, sqrt_d=0.1, head=1, dropout=dropout):
        super(p_seq_noise, self).__init__(input_dim, z_dim, output_dim, head, topk=topk, dropout=dropout)
        self.coo_net = coo_net
        self.noise_std = std
        self.sqrt_d = sqrt_d

    def forward(self, t, x):
        coo = x[:, :3]
        drift_coo = self.coo_net(t, coo)
        with torch.no_grad():
            mu_coo = x[:, :3]
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(x.device)
            edge_index = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(x.device)
        x = self.tfc_1(x, edge_index)
        x = self.silu(x)
        x = self.tfc_2(x, edge_index) + x
        x = self.silu(x)
        drift_seq = self.tfc_3(x, edge_index)

        noise = torch.randn_like(drift_seq) * self.noise_std * self.sqrt_d
        drift_seq += noise

        drift_ = torch.cat([drift_coo, drift_seq], dim=1)
        return drift_
