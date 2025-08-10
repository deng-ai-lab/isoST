
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from numpy import sqrt
from .base import init_optimizer, Trainer
from .utils.schedulers import ExponentialScheduler
from .utils.modules_ODE import PCooStep, PUnetStep, PJointV2
from .utils.losses import SinkhornLoss, one_nn_mse


class IsoST(Trainer):
    def __init__(self, gene_dim, hidden_dim, delta_d=0.2,
                 std_x=1, std_y=1, std_z=1, std_seq=1, stride=3,
                 alpha=0.1, warm_up_rate=0.01, head_num=1, K=5,
                 optimizer_name='Adam', method='euler',
                 lr=1e-4, weight_decay=1e-8,
                 beta_start_value=1, beta_end_value=1e-3,
                 beta_n_iterations=1000, beta_start_iteration=5000,
                 dual=True, **params):
        super().__init__(**params)
        self.dual = dual
        self.use_image_reg = False
        self.use_expr_reg = False
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.delta_d = delta_d
        self.sqrt_dd = sqrt(self.delta_d)
        self.std_x, self.std_y, self.std_z = std_x, std_y, std_z
        self.std_seq = std_seq
        self.w_coo = 1
        self.w_seq = alpha
        self.warm_up_rate = warm_up_rate
        self.device = self.get_device()

        self._init_modules()
        self._init_optimizer(lr, weight_decay, optimizer_name)
        self.method = method
        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations,
                                                   start_iteration=beta_start_iteration)

    def _init_modules(self):
        self.f_coo = PCooStep(3, self.hidden_dim, 3, self.std_x, self.std_y, self.std_z, self.sqrt_dd)
        self.g_coo = PCooStep(3, self.hidden_dim, 3, self.std_x, self.std_y, self.std_z, self.sqrt_dd)
        self.f_seq = PUnetStep(in_channels=3 + self.gene_dim, hidden_channels=self.hidden_dim,
                               out_channels=self.gene_dim, std=self.std_seq, sqrt_d=self.sqrt_dd)
        self.g_seq = PUnetStep(in_channels=3 + self.gene_dim, hidden_channels=self.hidden_dim,
                               out_channels=self.gene_dim, std=self.std_seq, sqrt_d=self.sqrt_dd)
        self.f_joint = PJointV2(self.f_coo, self.f_seq, topk=5)
        self.g_joint = PJointV2(self.g_coo, self.g_seq, topk=5)

    def _init_optimizer(self, lr, weight_decay, optimizer_name):
        self.opt = init_optimizer(optimizer_name, [
            {'params': self.f_joint.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.g_joint.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        ])

    def _get_items_to_store(self):
        return {
            'f_joint': self.f_joint.state_dict(),
            'g_joint': self.g_joint.state_dict(),
        }

    def freeze_coo_parameters(self):
        pass

    def freeze_seq_parameters(self):
        pass 

    def unfreeze_coo_parameters(self):
        pass

    def unfreeze_seq_parameters(self):
        pass

    def _compute_depth_dir(self, u_lists):
        depth_dir = []
        for i in range(len(u_lists) - 1):
            d0, d1 = torch.mean(u_lists[i][:, 2]), torch.mean(u_lists[i + 1][:, 2])
            NN = round(torch.abs(d1 - d0).item() / self.delta_d) + 1
            depth_dir.append(torch.linspace(d0, d1, steps=NN).to(self.device))
        self.depth_dir = depth_dir
        self.r_depth_dir = depth_dir[::-1]

    def _compute_feature_trajectory(self, model, u0, depth_target):
        return odeint(model, u0, depth_target, method=self.method)

    def _compute_loss_component(self, pred, true, loss_type):
        if loss_type == 'coo':
            return self.warm_up_rate * SinkhornLoss(pred[:, :3], true[:, :3])
        elif loss_type == 'seq':
            return self.warm_up_rate * one_nn_mse(pred[:, :3], pred[:, 3:], true[:, :3], true[:, 3:])
        else:
            coo_loss = SinkhornLoss(pred[:, :3], true[:, :3])
            seq_loss = one_nn_mse(pred[:, :3], pred[:, 3:], true[:, :3], true[:, 3:])
            return self.w_coo * coo_loss + self.w_seq * seq_loss

    def _compute_total_loss(self, u_lists, depth_dir, model, loss_type, reversed=False):
        total_loss = 0.0
        slide_num = len(u_lists)
        for kk in range(slide_num - self.stride):
            u0 = u_lists[kk]
            index_target, depth_target = [], None
            for s in range(self.stride):
                d = depth_dir[kk + s]
                if reversed:
                    d = d.flip(dims=[0])
                depth_target = d if s == 0 else torch.cat([depth_target[:-1], d], dim=0)
                index_target.append(len(depth_target) - 1 if s == 0 else index_target[-1] + len(d) - 1)

            trajectory = self._compute_feature_trajectory(model, u0, depth_target)
            for j, true in enumerate(u_lists[kk + 1: kk + self.stride + 1]):
                pred = trajectory[index_target[j]]
                loss = self._compute_loss_component(pred.to(self.device), true.to(self.device), loss_type)
                total_loss += loss / self.stride
        return total_loss

    def _compute_loss(self, u_lists, loss_type='joint'):
        total_loss = self._compute_total_loss(u_lists, self.depth_dir, self.f_joint, loss_type)
        if self.dual:
            r_u_lists = u_lists[::-1]
            total_loss += self._compute_total_loss(r_u_lists, self.r_depth_dir, self.g_joint, loss_type, reversed=True)
        beta = self.beta_scheduler(self.iterations)
        self._add_loss_item('beta', beta)
        self._add_loss_item('total loss', total_loss.item())
        return beta * total_loss

    def _train_step(self, u_lists, flag=None):
        self._compute_depth_dir(u_lists)
        loss = self._compute_loss(u_lists, loss_type=flag)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def fine_infer(self, data_dir, slide_names, mode, defined_d, result_dir, batch_num, device):
        from tqdm import tqdm
        import torch
        import numpy as np
        self.eval()
        with torch.no_grad():
            depth_known = []
            for i in tqdm(range(len(slide_names))):
                u = torch.load(f'{data_dir}/shuffled_{slide_names[i]}.pt').float().to(device)
                depth_known.append(torch.mean(u[:, 2]))
            depth_known = torch.tensor(depth_known).float().to(device)

            count = 0
            for i in tqdm(range(len(slide_names) - 1)):
                NN = round(torch.abs(depth_known[i] - depth_known[i + 1]).item() / defined_d) + 1
                depth_target = torch.linspace(depth_known[i], depth_known[i + 1], steps=NN, device=self.device)
                u = torch.load(f'{data_dir}/shuffled_{slide_names[i]}.pt').float().to(device)
                np.save(f'{result_dir}/{count}_forward.npy', u.detach().cpu().numpy())
                count += 1

                for b in range(batch_num):
                    num1 = round(len(u) / batch_num)
                    if b != batch_num - 1:
                        u_ = u[b * num1:(b + 1) * num1]
                    else:
                        u_ = u[b * num1:]
                    feature_trajectory_b = odeint(self.f_joint, u_, depth_target, method=self.method)
                    if b == 0:
                        feature_trajectory = feature_trajectory_b
                    else:
                        feature_trajectory = torch.cat([feature_trajectory, feature_trajectory_b], dim=1)

                if self.dual:
                    u_r = torch.load(f'{data_dir}/shuffled_{slide_names[i + 1]}.pt').float().to(device)
                    r_depth_target = depth_target.flip(dims=[0])
                    for b in range(batch_num):
                        num1 = round(len(u_r) / batch_num)
                        if b != batch_num - 1:
                            u_r_ = u_r[b * num1:(b + 1) * num1]
                        else:
                            u_r_ = u_r[b * num1:]
                        feature_trajectory_b_r = odeint(self.g_joint, u_r_, r_depth_target, method=self.method)
                        if b == 0:
                            feature_trajectory_r = feature_trajectory_b_r
                        else:
                            feature_trajectory_r = torch.cat([feature_trajectory_r, feature_trajectory_b_r], dim=1)
                    feature_trajectory_r = feature_trajectory_r.flip(dims=[0])
                    for kk in range(1, NN - 1):
                        result_forward = feature_trajectory[kk].detach().cpu().numpy()
                        result_backward = feature_trajectory_r[kk].detach().cpu().numpy()
                        if self.std_z <= 1e-4:
                            depth_target_numpy = depth_target.detach().cpu().numpy()
                            result_forward[:, 2] = depth_target_numpy[kk]
                            result_backward[:, 2] = depth_target_numpy[kk]
                        result = np.vstack([result_forward, result_backward])
                        np.save(f'{result_dir}/{count}_forward.npy', result)
                        count += 1
                else:
                    for kk in range(1, NN - 1):
                        result_forward = feature_trajectory[kk].detach().cpu().numpy()
                        if self.std_z <= 1e-4:
                            depth_target_numpy = depth_target.detach().cpu().numpy()
                            result_forward[:, 2] = depth_target_numpy[kk]
                        result = result_forward
                        np.save(f'{result_dir}/{count}_forward.npy', result)
                        count += 1
            u_last = torch.load(f'{data_dir}/shuffled_{slide_names[-1]}.pt').float().to(device)
            np.save(f'{result_dir}/{count}_forward.npy', u_last.detach().cpu().numpy())
            count += 1
        print('Done')
