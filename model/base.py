import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

import torch.optim as optimizer_module
from torch_cluster import knn_graph

# utility function to initialize an optimizer from its name
def init_optimizer(optimizer_name, params):
    assert hasattr(optimizer_module, optimizer_name)
    OptimizerClass = getattr(optimizer_module, optimizer_name)
    return OptimizerClass(params)

##########################
# Generic model class #
##########################
class Trainer(nn.Module):
    def __init__(self, device, log_loss_every=100, writer=None):
        super(Trainer, self).__init__()
        self.iterations = 0
        self.writer = writer
        self.log_loss_every = log_loss_every
        self.loss_items = {}
        self.device = device

    def get_device(self):
        return self.device

    def train_step(self, dataset, topk=5, flag=None):
        # Set all the models in model mode
        self.train(True)

        u_lists = dataset
        # Log the values in loss_items every log_loss_every iterations
        if not (self.writer is None):
            if (self.iterations + 1) % self.log_loss_every == 0:
                self._log_loss()

        # Move the data to the appropriate device
        device = self.get_device()
        # Perform the model step and update the iteration count
        self._train_step(u_lists, flag)
        self.iterations += 1

    def _add_loss_item(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, float) or isinstance(value, int)

        if not (name in self.loss_items):
            self.loss_items[name] = []

        self.loss_items[name].append(value)

    def _log_loss(self):
        # Log the expected value of the items in loss_items
        for key, values in self.loss_items.items():
            self.writer.add_scalar(tag=key, scalar_value=np.mean(values), global_step=self.iterations)
            self.loss_items[key] = []

    def save(self, model_path):
        items_to_save = self._get_items_to_store()
        items_to_save['iterations'] = self.iterations

        # Save the model and increment the checkpoint count
        torch.save(items_to_save, model_path)

    def load(self, model_path):
        items_to_load = torch.load(model_path)
        for key, value in items_to_load.items():
            assert hasattr(self, key)
            attribute = getattr(self, key)

            # Load the state dictionary for the stored modules and optimizers
            if isinstance(attribute, nn.Module) or isinstance(attribute, Optimizer):
                attribute.load_state_dict(value)

                # Move the optimizer parameters to the same correct device.
                # see https://github.com/pytorch/pytorch/issues/2830 for further details
                if isinstance(attribute, Optimizer):
                    device = list(value['state'].values())[0]['exp_avg'].device # Hack to identify the device
                    for state in attribute.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)

            # Otherwise just copy the value
            else:
                setattr(self, key, value)

    def _forward_coo(self, mu_coo, z, prior, func):
        with torch.no_grad():
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(self.device)
            edge = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(self.device)
        z[:, 2] = 0
        u_coo = mu_coo + self.std_coo * self.sqrt_dd * z
        drift_coo = func(u_coo, edge, prior)
        return drift_coo

    def _forward_seq(self, mu_coo, mu_seq, z, prior_seq, func_seq):
        with torch.no_grad():
            spatial_coo = mu_coo
            batch = torch.tensor([0] * len(spatial_coo)).to(self.device)
            edge = knn_graph(spatial_coo, self.topk, batch=batch, loop=False).to(self.device)
        z[:, 2] = 0
        with torch.no_grad():
            u_coo = mu_coo + self.std_coo * self.sqrt_dd * z[:, :3]
        u_seq = mu_seq + self.std_seq * self.sqrt_dd * z[:, 3:]
        drift_seq = func_seq(u_coo, u_seq, edge, prior_seq)
        return drift_seq

    def _f_coo(self, mu_coo, z, prior):
        return self._forward_coo(mu_coo, z, prior, self.f_coo)

    def _g_coo(self, mu_coo, z, prior):
        return self._forward_coo(mu_coo, z, prior, self.g_coo)

    def _f_seq(self, mu_coo, mu_seq, z, prior_coo, prior_seq):
        with torch.no_grad():
            drift_coo = self._f_coo(mu_coo, z[:, :3], prior_coo)
        drift_seq = self._forward_seq(mu_coo, mu_seq, z, prior_seq, self.f_seq)
        return drift_coo, drift_seq

    def _g_seq(self, mu_coo, mu_seq, z, prior_coo, prior_seq):
        with torch.no_grad():
            drift_coo = self._g_coo(mu_coo, z[:, :3], prior_coo)
        drift_seq = self._forward_seq(mu_coo, mu_seq, z, prior_seq, self.g_seq)
        return drift_coo, drift_seq

    def _f(self, mu_coo, mu_seq, z, prior_coo, prior_seq):
        # z[:, 2] = 0
        # u_coo = mu_coo + self.std_coo * self.sqrt_dd * z[:, :3]
        # u_seq = mu_seq + self.std_seq * self.sqrt_dd * z[:, 3:]

        drift_coo = self._f_coo(mu_coo, z[:, :3], prior_coo)
        drift_seq = self._forward_seq(mu_coo, mu_seq, z, prior_seq, self.f_seq)
        return drift_coo, drift_seq

    def _g(self, mu_coo, mu_seq, z, prior_coo, prior_seq):
        # z[:, 2] = 0
        # u_coo = mu_coo + self.std_coo * self.sqrt_dd * z[:, :3]
        # u_seq = mu_seq + self.std_seq * self.sqrt_dd * z[:, 3:]
        drift_coo = self._g_coo(mu_coo, z[:, :3], prior_coo)
        drift_seq = self._forward_seq(mu_coo, mu_seq, z, prior_seq, self.g_seq)
        return drift_coo, drift_seq

    def base_infer(self, u_1, u_2, edge_1, depth, mode, func, refer_slide=None):
        with torch.no_grad():
            self.eval()
            # forward
            if refer_slide == None:
                refer = u_2
            else:
                refer = refer_slide
            BiG21 = create_BiG(u_2, u_1, mode)
            BiG12 = create_BiG(u_1, u_2, mode)

            zv_1 = torch.mean(u_1[:, 2])
            n_i = abs(zv_1 - depth) / self.delta_d
            N_i = round(n_i.item())
            mu_f_coo, mu_f_seq = u_1[:, :3], u_1[:, 3:]
            # z = torch.zeros(u_1.shape[0], u_1.shape[1]).to(u_1.device)
            z = torch.randn(u_1.shape[0], u_1.shape[1]).to(u_1.device)
            coo_f_prior, seq_f_prior = self._get_prior(u_1, refer, None, BiG12, mode='joint')
            for j in range(1, N_i + 1):
                drift_f_coo, drift_f_seq = func(mu_f_coo, mu_f_seq, z, coo_f_prior, seq_f_prior)
                mu_f_coo = mu_f_coo + drift_f_coo * self.delta_d
                mu_f_seq = mu_f_seq + drift_f_seq * self.delta_d
            mu_f = torch.cat([mu_f_coo, mu_f_seq], dim=1)
            mu_2 = torch.sparse.mm(BiG21, mu_f)
            # mu_2 = mu_plus_j
        return mu_2.detach()

    def _infer(self, u_1, edge_1, u_2, edge_2, depth, mode, forward_func, backward_func, refer_slide1=None, refer_slide2=None):
        with torch.no_grad():
            self.eval()

            mu_plus_j = forward_func(u_1, u_2, edge_1, depth, mode, refer_slide1)
            mu_minus_j = backward_func(u_2, u_1, edge_2, depth, mode, refer_slide2)

            combined = torch.cat((mu_plus_j, mu_minus_j), dim=0)

            shuffled_indices = torch.randperm(combined.size(0))
            combined_shuffled = combined[shuffled_indices]

            desired_size = (mu_plus_j.shape[0] + mu_minus_j.shape[0]) // 2
            sampled_data = combined_shuffled[:desired_size]
        return sampled_data.detach()

    def _infer_drift(self, u_1, u_2, edge_1, depth, mode, func, refer_slide=None):
        with torch.no_grad():
            self.eval()
            # forward
            if refer_slide == None:
                refer = u_2
            else:
                refer = refer_slide
            BiG21 = create_BiG(u_2[:, :2], u_1[:, :2])
            #
            zv_1 = torch.mean(u_1[:, 2])
            n_i = abs(zv_1 - depth) / self.delta_d
            N_i = round(n_i.item())
            mu_f_coo = u_1[:, :3]
            drift_f_coo = torch.zeros(u_2.shape[0], 3).to(u_1.device)
            z = torch.zeros(u_1.shape[0], 3).to(u_1.device)
            # z = torch.randn(u_1.shape[0], u_1.shape[1]).to(u_1.device)
            for j in range(1, N_i + 1):
                drift_f_coo = self._forward_coo(mu_f_coo, edge_1, z, func)
                mu_f_coo = mu_f_coo + drift_f_coo * self.delta_d
            mu_f = drift_f_coo
            mu_2 = torch.sparse.mm(BiG21, mu_f)

        return mu_2.detach()

    def forward_infer(self, u_2, u_1, edge_2, depth, mode, refer_slide=None):
        return self.base_infer(u_2, u_1, edge_2, depth, mode, self._f, refer_slide)

    def backward_infer(self, u_2, u_1, edge_2, depth, mode, refer_slide=None):
        return self.base_infer(u_2, u_1, edge_2, depth, mode, self._g, refer_slide)

    def forward_infer_drift(self, u_1, u_2, edge_1, depth, mode, refer_slide=None):
        return self._infer_drift(u_1, u_2, edge_1, depth, mode, self.f_coo, refer_slide)

    def backward_infer_drift(self, u_2, u_1, edge_2, depth, mode, refer_slide=None):
        return self._infer_drift(u_2, u_1, edge_2, depth, mode, self.g_coo, refer_slide)

    def infer(self, u_1, edge_1, u_2, edge_2, depth, mode, refer_slide1=None, refer_slide2=None):
        return self._infer(u_1, edge_1, u_2, edge_2,
                           depth, mode,
                           self.forward_infer, self.backward_infer,
                           refer_slide1, refer_slide2)

    def infer_drift(self, u_1, edge_1, u_2, edge_2, depth, mode, refer_slide1=None, refer_slide2=None):
        return self._infer(u_1, edge_1, u_2, edge_2,
                           depth, mode, self.forward_infer_drift, self.backward_infer_drift,
                           refer_slide1, refer_slide2)


    def total_infer(self, data_dir, slide_names, mode, result_dir, batch_num, device):
        import numpy as np
        self.eval()
        with torch.no_grad():
            for b in range(batch_num):
                u_lists = []
                slide_num = len(slide_names)
                print(f'===There are {slide_num} slides used for inference.===')
                for i in slide_names:
                    u = torch.load(f'{data_dir}/shuffled_{i}.pt').float().to(device)
                    num1 = round(len(u) / batch_num)
                    if b != batch_num - 1:
                        u_ = u[b * num1:(b + 1) * num1]
                    else:
                        u_ = u[b * num1:]
                    u_lists.append(u_)

                BiG21_list = []
                BiG12_list = []
                # print(f'===Generating bipartite garphs. There are 2x{slide_num-1} bipartite graphs.===')
                for i in range(len(slide_names) - 1):
                    # print(f'Calculating the bipartite graph between the {i+1}-th and {i+2}-th slices.')
                    idx1 = slide_names[i]
                    idx2 = slide_names[i + 1]
                    BiG21 = torch.load(f'{data_dir}/BiG_{mode}_{batch_num}/BiG_{idx2}_{idx1}_{b}.pt').to(device)
                    BiG12 = torch.load(f'{data_dir}/BiG_{mode}_{batch_num}/BiG_{idx1}_{idx2}_{b}.pt').to(device)
                    BiG21_list.append(BiG21)
                    BiG12_list.append(BiG12)
                print("Graph Loaded")

                slide_num = len(u_lists)
                all_index = []
                for slide in u_lists:
                    all_index.append(torch.mean(slide[:, 2]))

                coo_f_prior_list = []
                seq_f_prior_list = []
                coo_g_prior_list = []
                seq_g_prior_list = []
                u_lists_r = list(reversed(u_lists))
                for i in range(slide_num - 1):
                    u_1 = u_lists[i]
                    u_2 = u_lists[i+1]
                    u_1_r = u_lists_r[i]
                    u_2_r = u_lists_r[i+1]
                    BiG1s = BiG12_list[i]
                    BiGs1 = BiG21_list[slide_num-i-2]

                    coo_f_prior, seq_f_prior = self._get_prior(u_1, u_2, None, BiG1s, mode='joint')
                    coo_g_prior, seq_g_prior = self._get_prior(u_1_r, u_2_r, None, BiGs1, mode='joint')

                    coo_f_prior_list.append(coo_f_prior)
                    seq_f_prior_list.append(seq_f_prior)
                    coo_g_prior_list.append(coo_g_prior)
                    seq_g_prior_list.append(seq_g_prior)


                # forward
                count_f = 0
                for kk in range(slide_num - 1):
                    coo_f_prior, seq_f_prior = coo_f_prior_list[kk],  seq_f_prior_list[kk]
                    u_1 = u_lists[kk]
                    np.save(f'{result_dir}/{count_f}_truth_{kk}_{batch_num}_{b}.npy', u_1.detach().cpu().numpy())
                    count_f += 1
                    u_2 = u_lists[kk + self.stride]
                    zv_i1 = torch.mean(u_1[:, 2])
                    zv_i2 = torch.mean(u_2[:, 2])
                    u1_coo, u1_seq = u_1[:, :3], u_1[:, 3:]
                    mu_f_coo, mu_f_seq = u1_coo, u1_seq
                    n_i = abs(zv_i2 - zv_i1) / self.delta_d
                    N_i = round(n_i.item())
                    for k in range(N_i - 1):
                        z_f = torch.zeros(u_1.shape[0], u_1.shape[1])
                        z_f = z_f.to(self.device)
                        drift_f_coo, drift_f_seq = self._f(mu_f_coo, mu_f_seq, z_f, coo_f_prior, seq_f_prior)
                        mu_f_coo = mu_f_coo + drift_f_coo * self.delta_d
                        mu_f_seq = mu_f_seq + drift_f_seq * self.delta_d
                        mu_f = torch.cat([mu_f_coo, mu_f_seq], dim=1).detach()
                        np.save(f'{result_dir}/{count_f}_forward_{kk}_{batch_num}_{b}.npy', mu_f.cpu().numpy())
                        count_f += 1
                np.save(f'{result_dir}/{count_f}_truth_{kk+1}_{batch_num}_{b}.npy', u_2.detach().cpu().numpy())
                count_f += 1

                # backward
                count_g = 0
                for kk in range(slide_num - 1):
                    coo_g_prior, seq_g_prior = coo_g_prior_list[kk], seq_g_prior_list[kk]
                    u_1 = u_lists_r[kk]
                    count_g += 1
                    u_2 = u_lists_r[kk + 1]
                    zv_i1 = torch.mean(u_1[:, 2])
                    zv_i2 = torch.mean(u_2[:, 2])
                    u1_coo, u1_seq = u_1[:, :3], u_1[:, 3:]
                    mu_g_coo, mu_g_seq = u1_coo, u1_seq
                    n_i = abs(zv_i2 - zv_i1) / self.delta_d
                    N_i = round(n_i.item())
                    for k in range(N_i - 1):
                        z_f = torch.zeros(u_1.shape[0], u_1.shape[1])
                        z_f = z_f.to(self.device)
                        drift_g_coo, drift_g_seq = self._g(mu_g_coo, mu_g_seq, z_f, coo_g_prior, seq_g_prior)
                        mu_g_coo = mu_g_coo + drift_g_coo * self.delta_d
                        mu_g_seq = mu_g_seq + drift_g_seq * self.delta_d
                        mu_g = torch.cat([mu_g_coo, mu_g_seq], dim=1).detach()
                        np.save(f'{result_dir}/{count_f - count_g - 1}_backward_{kk}_{batch_num}_{b}.npy',
                                mu_g.cpu().numpy())
                        count_g += 1
            count_g += 1
            print('Done')

    def fine_infer(self, data_dir, slide_names, mode, defined_d, result_dir, batch_num, device):
        from tqdm import tqdm
        import torch
        import numpy as np
        from multiprocessing import Pool, cpu_count
        self.eval()
        with torch.no_grad():
            for b in range(batch_num):
                u_lists = []
                slide_num = len(slide_names)
                print(f'===There are {slide_num} slides used for inference.===')
                for i in slide_names:
                    u = torch.load(f'{data_dir}/shuffled_{i}.pt').float().to(device)
                    num1 = round(len(u) / batch_num)
                    if b != batch_num - 1:
                        u_ = u[b * num1:(b + 1) * num1]
                    else:
                        u_ = u[b * num1:]
                    u_lists.append(u_)

                BiG21_list = []
                BiG12_list = []
                # print(f'===Generating bipartite garphs. There are 2x{slide_num-1} bipartite graphs.===')
                for i in range(len(slide_names) - 1):
                    # print(f'Calculating the bipartite graph between the {i+1}-th and {i+2}-th slices.')
                    idx1 = slide_names[i]
                    idx2 = slide_names[i + 1]
                    BiG21 = torch.load(f'{data_dir}/BiG_{mode}_{batch_num}/BiG_{idx2}_{idx1}_{b}.pt').to(device)
                    BiG12 = torch.load(f'{data_dir}/BiG_{mode}_{batch_num}/BiG_{idx1}_{idx2}_{b}.pt').to(device)
                    BiG21_list.append(BiG21)
                    BiG12_list.append(BiG12)
                print("Graph Loaded")

                slide_num = len(u_lists)
                all_index = []
                for slide in u_lists:
                    all_index.append(torch.mean(slide[:, 2]))

                coo_f_prior_list = []
                seq_f_prior_list = []
                coo_g_prior_list = []
                seq_g_prior_list = []
                u_lists_r = list(reversed(u_lists))
                for i in range(slide_num - 1):
                    u_1 = u_lists[i]
                    u_2 = u_lists[i+1]
                    u_1_r = u_lists_r[i]
                    u_2_r = u_lists_r[i+1]
                    BiG1s = BiG12_list[i]
                    BiGs1 = BiG21_list[slide_num-i-2]

                    coo_f_prior, seq_f_prior = self._get_prior(u_1, u_2, None, BiG1s, mode='joint')
                    coo_g_prior, seq_g_prior = self._get_prior(u_1_r, u_2_r, None, BiGs1, mode='joint')

                    coo_f_prior_list.append(coo_f_prior)
                    seq_f_prior_list.append(seq_f_prior)
                    coo_g_prior_list.append(coo_g_prior)
                    seq_g_prior_list.append(seq_g_prior)


                # forward
                count_f = 0
                for kk in tqdm(range(slide_num - 1)):
                    coo_f_prior, seq_f_prior = coo_f_prior_list[kk],  seq_f_prior_list[kk]
                    u_1 = u_lists[kk]
                    np.save(f'{result_dir}/{count_f}_truth_{kk}_{batch_num}_{b}.npy', u_1.detach().cpu().numpy())
                    count_f += 1
                    u_2 = u_lists[kk + self.stride]
                    zv_i1 = torch.mean(u_1[:, 2])
                    zv_i2 = torch.mean(u_2[:, 2])
                    u1_coo, u1_seq = u_1[:, :3], u_1[:, 3:]
                    mu_f_coo, mu_f_seq = u1_coo, u1_seq
                    n_i = abs(zv_i2 - zv_i1) / defined_d
                    N_i = round(n_i.item())
                    for k in range(N_i - 1):
                        z_f = torch.zeros(u_1.shape[0], u_1.shape[1])
                        z_f = z_f.to(self.device)
                        drift_f_coo, drift_f_seq = self._f(mu_f_coo, mu_f_seq, z_f, coo_f_prior, seq_f_prior)
                        mu_f_coo = mu_f_coo + drift_f_coo * defined_d
                        mu_f_seq = mu_f_seq + drift_f_seq * defined_d
                        mu_f = torch.cat([mu_f_coo, mu_f_seq], dim=1).detach()
                        np.save(f'{result_dir}/{count_f}_forward_{kk}_{batch_num}_{b}.npy', mu_f.cpu().numpy())
                        count_f += 1
                np.save(f'{result_dir}/{count_f}_truth_{kk+1}_{batch_num}_{b}.npy', u_2.detach().cpu().numpy())
                count_f += 1

                # backward
                count_g = 0
                for kk in tqdm(range(slide_num - 1)):
                    coo_g_prior, seq_g_prior = coo_g_prior_list[kk], seq_g_prior_list[kk]
                    u_1 = u_lists_r[kk]
                    count_g += 1
                    u_2 = u_lists_r[kk + 1]
                    zv_i1 = torch.mean(u_1[:, 2])
                    zv_i2 = torch.mean(u_2[:, 2])
                    u1_coo, u1_seq = u_1[:, :3], u_1[:, 3:]
                    mu_g_coo, mu_g_seq = u1_coo, u1_seq
                    n_i = abs(zv_i2 - zv_i1) / defined_d
                    N_i = round(n_i.item())
                    for k in range(N_i - 1):
                        z_f = torch.zeros(u_1.shape[0], u_1.shape[1])
                        z_f = z_f.to(self.device)
                        drift_g_coo, drift_g_seq = self._g(mu_g_coo, mu_g_seq, z_f, coo_g_prior, seq_g_prior)
                        mu_g_coo = mu_g_coo + drift_g_coo * defined_d
                        mu_g_seq = mu_g_seq + drift_g_seq * defined_d
                        mu_g = torch.cat([mu_g_coo, mu_g_seq], dim=1).detach()
                        np.save(f'{result_dir}/{count_f - count_g - 1}_backward_{kk}_{batch_num}_{b}.npy',
                                mu_g.cpu().numpy())
                        count_g += 1
            count_g += 1
            print('Done')

    def _get_items_to_store(self):
        return dict()

    def _train_step(self, st_data, sub_data):
        raise NotImplemented()
