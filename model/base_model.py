
import csv
import os

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

    def fine_infer(
            self,
            data_dir,
            slide_names,
            mode,
            defined_d,
            result_dir,
            batch_num,
            device,
            start_count=0,
            required_targets=None,
            target_output_rows=None,
    ):
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

            count = int(start_count)
            for i in tqdm(range(len(slide_names) - 1)):
                depth_start = float(depth_known[i].item())
                depth_end = float(depth_known[i + 1].item())
                increasing_depth = depth_start < depth_end
                interval_targets = sorted(
                    [
                        target
                        for target in (required_targets or [])
                        if (
                            depth_start < float(target[2]) < depth_end
                            if increasing_depth
                            else depth_end < float(target[2]) < depth_start
                        )
                    ],
                    key=lambda target: float(target[2]),
                    reverse=not increasing_depth,
                )
                interval_target_index = {}
                if interval_targets:
                    depth_target, interval_target_index = self._build_edge_depth_grid(
                        depth_start,
                        [*interval_targets, (None, depth_end, depth_end)],
                        defined_d,
                        self.device,
                    )
                    interval_target_index = {
                        grid_index: target
                        for grid_index, target in interval_target_index.items()
                        if target[0] is not None
                    }
                    NN = len(depth_target)
                else:
                    NN = round(abs(depth_start - depth_end) / defined_d) + 1
                    depth_target = torch.linspace(
                        depth_start,
                        depth_end,
                        steps=NN,
                        device=self.device,
                    )
                u = torch.load(f'{data_dir}/shuffled_{slide_names[i]}.pt').float().to(device)
                interval_start_count = count
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

                if target_output_rows is not None:
                    for grid_index, target in sorted(interval_target_index.items()):
                        target_output_rows.append({
                            'target_slide_name': target[0],
                            'region': 'interpolation',
                            'target_nominal_z': float(target[1]),
                            'target_model_z': float(target[2]),
                            'output_index': interval_start_count + grid_index,
                            'relative_path': (
                                f'{interval_start_count + grid_index}_forward.npy'
                            ),
                        })
            u_last = torch.load(f'{data_dir}/shuffled_{slide_names[-1]}.pt').float().to(device)
            np.save(f'{result_dir}/{count}_forward.npy', u_last.detach().cpu().numpy())
            count += 1
        print('Done')
        return count

    @staticmethod
    def _build_edge_depth_grid(anchor_z, ordered_targets, defined_d, device):
        """Build a piecewise Euler grid that lands exactly on every target z."""
        if defined_d <= 0:
            raise ValueError('defined_d must be positive.')

        pieces = [torch.tensor([float(anchor_z)], dtype=torch.float32, device=device)]
        target_index = {}
        cursor = float(anchor_z)
        grid_index = 0

        for target_name, target_nominal_z, target_model_z in ordered_targets:
            target_model_z = float(target_model_z)
            distance = abs(target_model_z - cursor)
            if distance <= 1e-12:
                raise ValueError(
                    f'Target {target_name!r} has the same z as the preceding point ({cursor}).'
                )

            steps = max(round(distance / defined_d) + 1, 2)
            segment = torch.linspace(
                cursor,
                target_model_z,
                steps=steps,
                dtype=torch.float32,
                device=device,
            )
            pieces.append(segment[1:])
            grid_index += steps - 1
            target_index[grid_index] = (
                target_name,
                float(target_nominal_z),
                target_model_z,
            )
            cursor = target_model_z

        return torch.cat(pieces), target_index

    def _integrate_edge_in_batches(self, model, anchor, depth_target, batch_num):
        if batch_num < 1:
            raise ValueError('batch_num must be at least 1.')
        if batch_num > len(anchor):
            raise ValueError(
                f'batch_num={batch_num} exceeds the anchor cell count ({len(anchor)}).'
            )

        trajectories = []
        for batch_index in range(batch_num):
            batch_size = round(len(anchor) / batch_num)
            if batch_index != batch_num - 1:
                anchor_batch = anchor[
                    batch_index * batch_size:(batch_index + 1) * batch_size
                ]
            else:
                anchor_batch = anchor[batch_index * batch_size:]

            topk = getattr(model, 'topk', 0)
            if len(anchor_batch) <= topk:
                raise ValueError(
                    f'Inference batch {batch_index} has {len(anchor_batch)} cells; '
                    f'the model requires more than topk={topk}.'
                )

            trajectories.append(
                odeint(model, anchor_batch, depth_target, method=self.method)
            )

        return torch.cat(trajectories, dim=1)

    def _save_edge_trajectory(
            self,
            trajectory,
            depth_target,
            target_index,
            result_dir,
            count,
            direction,
            anchor_name,
            anchor_z,
    ):
        import numpy as np

        if direction == 'left':
            # Integration runs from the boundary outwards (decreasing z), but files
            # remain globally ordered from the smallest z to the largest z.
            grid_indices = range(len(depth_target) - 1, 0, -1)
            model_name = 'g_joint'
        elif direction == 'right':
            grid_indices = range(1, len(depth_target))
            model_name = 'f_joint'
        else:
            raise ValueError(f'Unknown edge direction: {direction!r}')

        rows = []
        for grid_index in grid_indices:
            integration_z = float(depth_target[grid_index].item())
            result = trajectory[grid_index].detach().cpu().numpy()
            if self.std_z <= 1e-4:
                result[:, 2] = integration_z

            relative_path = f'{count}_forward.npy'
            np.save(os.path.join(result_dir, relative_path), result)
            requested_target = target_index.get(grid_index)
            rows.append({
                'relative_path': relative_path,
                'output_index': count,
                'direction': direction,
                'model': model_name,
                'anchor_slide_name': anchor_name,
                'anchor_z': float(anchor_z),
                'integration_z': integration_z,
                'is_requested_target': requested_target is not None,
                'target_slide_name': requested_target[0] if requested_target else '',
                'target_nominal_z': requested_target[1] if requested_target else '',
                'target_model_z': requested_target[2] if requested_target else '',
                'n_rows': int(result.shape[0]),
                'n_features': int(result.shape[1]),
            })
            count += 1

        return count, rows

    def edge_infer(
            self,
            data_dir,
            train_slide_names,
            train_z,
            target_slide_names,
            target_z,
            mode,
            defined_d,
            result_dir,
            batch_num,
            device,
    ):
        """Interpolate between anchors and extrapolate beyond both observed edges.

        ``train_z`` and ``target_z`` are public/nominal acquisition depths. Only
        training tensors are loaded. A monotonic piecewise-linear mapping from
        nominal depth to model mean-z is fitted on the training anchors, so no
        expression, xy, or z values from held-out tensors leak into extrapolation.
        """
        import numpy as np

        if len(train_slide_names) < 2:
            raise ValueError('At least two ordered training anchors are required.')
        if len(train_slide_names) != len(train_z):
            raise ValueError('train_slide_names and train_z must have equal length.')
        if len(target_slide_names) != len(target_z):
            raise ValueError('target_slide_names and target_z must have equal length.')
        if len(set(train_slide_names)) != len(train_slide_names):
            raise ValueError('Training anchor names must be unique.')
        if len(set(target_slide_names)) != len(target_slide_names):
            raise ValueError('Target slide names must be unique.')
        overlap = set(train_slide_names) & set(target_slide_names)
        if overlap:
            raise ValueError(f'Train/target overlap is not allowed: {sorted(overlap)}')
        if abs(float(defined_d) - float(self.delta_d)) > 1e-8:
            raise ValueError(
                f'defined_d={defined_d} must match the trained model delta_d={self.delta_d}.'
            )

        os.makedirs(result_dir, exist_ok=True)
        self.eval()

        with torch.no_grad():
            train_depths = []
            for slide_name in train_slide_names:
                anchor = torch.load(
                    os.path.join(data_dir, f'shuffled_{slide_name}.pt')
                ).float()
                train_depths.append(float(torch.mean(anchor[:, 2]).item()))

            if any(
                right <= left
                for left, right in zip(train_depths[:-1], train_depths[1:])
            ):
                raise ValueError(
                    'train_slide_names must be strictly increasing in tensor mean-z.'
                )

            left_anchor_z = train_depths[0]
            right_anchor_z = train_depths[-1]
            train_nominal_z = [float(z_value) for z_value in train_z]
            target_nominal_pairs = [
                (name, float(z_value)) for name, z_value in zip(target_slide_names, target_z)
            ]
            if not all(np.isfinite(z_value) for z_value in train_nominal_z):
                raise ValueError('All train_z values must be finite.')
            if not all(
                right > left
                for left, right in zip(train_nominal_z[:-1], train_nominal_z[1:])
            ):
                raise ValueError('train_z must be strictly increasing.')
            if not all(np.isfinite(z_value) for _, z_value in target_nominal_pairs):
                raise ValueError('All target_z values must be finite.')

            def nominal_to_model_z(nominal_z):
                nominal_z = float(nominal_z)
                if nominal_z < train_nominal_z[0]:
                    slope = (
                        (train_depths[1] - train_depths[0])
                        / (train_nominal_z[1] - train_nominal_z[0])
                    )
                    return train_depths[0] + slope * (nominal_z - train_nominal_z[0])
                if nominal_z > train_nominal_z[-1]:
                    slope = (
                        (train_depths[-1] - train_depths[-2])
                        / (train_nominal_z[-1] - train_nominal_z[-2])
                    )
                    return train_depths[-1] + slope * (nominal_z - train_nominal_z[-1])
                return float(np.interp(nominal_z, train_nominal_z, train_depths))

            target_pairs = [
                (name, nominal_z, nominal_to_model_z(nominal_z))
                for name, nominal_z in target_nominal_pairs
            ]

            left_targets = sorted(
                [pair for pair in target_pairs if pair[1] < train_nominal_z[0]],
                key=lambda pair: pair[1],
                reverse=True,
            )
            right_targets = sorted(
                [pair for pair in target_pairs if pair[1] > train_nominal_z[-1]],
                key=lambda pair: pair[1],
            )
            interior_targets = sorted(
                [
                    pair for pair in target_pairs
                    if train_nominal_z[0] < pair[1] < train_nominal_z[-1]
                ],
                key=lambda pair: pair[1],
            )
            if not left_targets and not right_targets:
                raise ValueError(
                    'No target lies outside the observed anchor range; use fine_infer instead.'
                )
            if left_targets and not self.dual:
                raise RuntimeError(
                    'Left extrapolation requires dual=True because it uses g_joint.'
                )

            count = 0
            manifest_rows = []
            target_output_rows = []

            if left_targets:
                left_anchor_name = train_slide_names[0]
                left_anchor = torch.load(
                    os.path.join(data_dir, f'shuffled_{left_anchor_name}.pt')
                ).float().to(device)
                left_grid, left_target_index = self._build_edge_depth_grid(
                    left_anchor_z,
                    left_targets,
                    defined_d,
                    device,
                )
                left_trajectory = self._integrate_edge_in_batches(
                    self.g_joint,
                    left_anchor,
                    left_grid,
                    batch_num,
                )
                count, rows = self._save_edge_trajectory(
                    left_trajectory,
                    left_grid,
                    left_target_index,
                    result_dir,
                    count,
                    'left',
                    left_anchor_name,
                    left_anchor_z,
                )
                manifest_rows.extend(rows)
                target_output_rows.extend([
                    {
                        'target_slide_name': row['target_slide_name'],
                        'region': 'left_extrapolation',
                        'target_nominal_z': row['target_nominal_z'],
                        'target_model_z': row['target_model_z'],
                        'output_index': row['output_index'],
                        'relative_path': row['relative_path'],
                    }
                    for row in rows
                    if row['is_requested_target']
                ])

            count = self.fine_infer(
                data_dir,
                train_slide_names,
                mode,
                defined_d,
                result_dir,
                batch_num,
                device,
                start_count=count,
                required_targets=interior_targets,
                target_output_rows=target_output_rows,
            )

            if right_targets:
                right_anchor_name = train_slide_names[-1]
                right_anchor = torch.load(
                    os.path.join(data_dir, f'shuffled_{right_anchor_name}.pt')
                ).float().to(device)
                right_grid, right_target_index = self._build_edge_depth_grid(
                    right_anchor_z,
                    right_targets,
                    defined_d,
                    device,
                )
                right_trajectory = self._integrate_edge_in_batches(
                    self.f_joint,
                    right_anchor,
                    right_grid,
                    batch_num,
                )
                count, rows = self._save_edge_trajectory(
                    right_trajectory,
                    right_grid,
                    right_target_index,
                    result_dir,
                    count,
                    'right',
                    right_anchor_name,
                    right_anchor_z,
                )
                manifest_rows.extend(rows)
                target_output_rows.extend([
                    {
                        'target_slide_name': row['target_slide_name'],
                        'region': 'right_extrapolation',
                        'target_nominal_z': row['target_nominal_z'],
                        'target_model_z': row['target_model_z'],
                        'output_index': row['output_index'],
                        'relative_path': row['relative_path'],
                    }
                    for row in rows
                    if row['is_requested_target']
                ])

        manifest_path = os.path.join(result_dir, 'edge_extrapolation_manifest.csv')
        fieldnames = [
            'relative_path',
            'output_index',
            'direction',
            'model',
            'anchor_slide_name',
            'anchor_z',
            'integration_z',
            'is_requested_target',
            'target_slide_name',
            'target_nominal_z',
            'target_model_z',
            'n_rows',
            'n_features',
        ]
        with open(manifest_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        mapping_path = os.path.join(result_dir, 'target_depth_mapping.csv')
        target_output_lookup = {
            row['target_slide_name']: row for row in target_output_rows
        }
        missing_target_outputs = set(target_slide_names) - set(target_output_lookup)
        if missing_target_outputs:
            raise RuntimeError(
                f'No exact output was recorded for target(s): {sorted(missing_target_outputs)}'
            )
        with open(mapping_path, 'w', newline='') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    'target_slide_name',
                    'region',
                    'target_nominal_z',
                    'target_model_z',
                    'output_index',
                    'relative_path',
                ],
            )
            writer.writeheader()
            writer.writerows(sorted(
                target_output_rows,
                key=lambda row: float(row['target_nominal_z']),
            ))

        print(
            f'Edge inference done: {len(left_targets)} left target(s), '
            f'{len(right_targets)} right target(s), {count} total output layer(s).'
        )
        return {
            'manifest_path': manifest_path,
            'target_depth_mapping_path': mapping_path,
            'n_left_targets': len(left_targets),
            'n_right_targets': len(right_targets),
            'n_output_layers': count,
        }
