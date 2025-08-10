import os
import yaml
import numpy as np
import model as training_module
import torch
from torch_cluster import knn_graph
import pandas as pd

def infer(
        experiment_dir,
        data_dir,
        u_1_name,
        u_2_name,
        topk,
        depth,
        result_dir,
        direction = 'biaxial',
        device='cuda'):
    config_file = None
    # Check if the experiment directory already contains a model
    pretrained = os.path.isfile(os.path.join(experiment_dir, 'model.pt')) \
                 and os.path.isfile(os.path.join(experiment_dir, 'config.yml'))

    resume_training = pretrained

    if resume_training:
        load_model_file = os.path.join(experiment_dir, 'model.pt')
        config_file = os.path.join(experiment_dir, 'config.yml')
    # Load the configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Copy it to the experiment folder
    with open(os.path.join(experiment_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)

    # Instantiating the trainer according to the specified configuration
    TrainerClass = getattr(training_module, config['trainer'])
    trainer = TrainerClass(device=device, **config['params'])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Resume the model if specified
    if load_model_file:
        trainer.load(load_model_file)
        print('Pretrained Model Loaded!')
    trainer.to(device)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #############
    # inference #
    #############
    u_1_np = pd.read_csv(f'{data_dir}/{u_1_name}.csv', index_col=0).values.astype(np.float32)
    u_2_np = pd.read_csv(f'{data_dir}/{u_2_name}.csv', index_col=0).values.astype(np.float32)
    u_1 = torch.as_tensor(u_1_np).to(device)
    u_2 = torch.as_tensor(u_2_np).to(device)

    spatial_coo_1 = u_1[:, :2]
    batch_1 = torch.tensor([0] * len(spatial_coo_1)).to(device)
    edge_1 = knn_graph(spatial_coo_1, topk, batch=batch_1, loop=False).to(device)

    spatial_coo_2 = u_2[:, :2]
    batch_2 = torch.tensor([0] * len(spatial_coo_2)).to(device)
    edge_2 = knn_graph(spatial_coo_2, topk, batch=batch_2, loop=False).to(device)

    if direction == 'forward':
        x_pred = trainer.forward_infer(u_1, u_2, edge_1, depth).cpu().numpy()
    elif direction == 'backward':
        x_pred = trainer.backward_infer(u_2, u_1, edge_2, depth).cpu().numpy()
    else:
        x_pred = trainer.infer(u_1, edge_1, u_2, edge_2, depth).cpu().numpy()
    return x_pred
    # np.save(result_dir + f'/{save_name}.npy', x_pred)

def total_inference(
        experiment_dir,
        data_dir,
        u_name_list,
        mode,
        result_dir,
        batch_num,
        device='cuda'):
    config_file = None
    # Check if the experiment directory already contains a model
    pretrained = os.path.isfile(os.path.join(experiment_dir, 'model.pt')) \
                 and os.path.isfile(os.path.join(experiment_dir, 'config.yml'))

    resume_training = pretrained

    if resume_training:
        load_model_file = os.path.join(experiment_dir, 'model.pt')
        config_file = os.path.join(experiment_dir, 'config.yml')
    # Load the configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Copy it to the experiment folder
    with open(os.path.join(experiment_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)

    # Instantiating the trainer according to the specified configuration
    TrainerClass = getattr(training_module, config['trainer'])
    trainer = TrainerClass(device=device, **config['params'])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Resume the model if specified
    if load_model_file:
        trainer.load(load_model_file)
        print('Pretrained Model Loaded!')
    trainer.to(device)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #############
    # inference #
    #############
    trainer.total_infer(data_dir, u_name_list, mode, result_dir, batch_num, device)


def fine_inference(
        experiment_dir,
        data_dir,
        u_name_list,
        mode,
        defined_d,
        result_dir,
        batch_num,
        device='cuda'):
    config_file = None
    # Check if the experiment directory already contains a model
    pretrained = os.path.isfile(os.path.join(experiment_dir, 'model.pt')) \
                 and os.path.isfile(os.path.join(experiment_dir, 'config.yml'))

    resume_training = pretrained

    if resume_training:
        load_model_file = os.path.join(experiment_dir, 'model.pt')
        config_file = os.path.join(experiment_dir, 'config.yml')
    # Load the configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Copy it to the experiment folder
    with open(os.path.join(experiment_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)

    # Instantiating the trainer according to the specified configuration
    TrainerClass = getattr(training_module, config['trainer'])
    trainer = TrainerClass(device=device, **config['params'])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Resume the model if specified
    if load_model_file:
        trainer.load(load_model_file)
        print('Pretrained Model Loaded!')
    trainer.to(device)

    #############
    # inference #
    #############
    trainer.fine_infer(data_dir, u_name_list, mode, defined_d, result_dir, batch_num, device)