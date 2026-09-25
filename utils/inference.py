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
        device='cuda',
        train_z=None,
        target_slide_names=None,
        target_z=None):
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
    mapping_inputs = (train_z, target_slide_names, target_z)
    if any(value is not None for value in mapping_inputs) and not all(
            value is not None for value in mapping_inputs):
        raise ValueError(
            'train_z, target_slide_names and target_z must be provided together.'
        )

    required_targets = None
    target_output_rows = None
    if all(value is not None for value in mapping_inputs):
        if len(u_name_list) != len(train_z):
            raise ValueError('u_name_list and train_z must have equal length.')
        if len(target_slide_names) != len(target_z):
            raise ValueError(
                'target_slide_names and target_z must have equal length.'
            )
        if len(set(target_slide_names)) != len(target_slide_names):
            raise ValueError('target_slide_names must be unique.')
        overlap = set(u_name_list) & set(target_slide_names)
        if overlap:
            raise ValueError(
                f'Train/target overlap is not allowed: {sorted(overlap)}'
            )

        train_z_array = np.asarray(train_z, dtype=float)
        target_z_array = np.asarray(target_z, dtype=float)
        if not np.all(np.isfinite(train_z_array)) or not np.all(
                np.isfinite(target_z_array)):
            raise ValueError('All train_z and target_z values must be finite.')
        if np.any(np.diff(train_z_array) <= 0):
            raise ValueError('train_z must be strictly increasing.')
        if np.any(target_z_array <= train_z_array[0]) or np.any(
                target_z_array >= train_z_array[-1]):
            raise ValueError(
                'fine_inference only accepts interpolation targets strictly '
                'inside the observed anchor range.'
            )

        # Derive the nominal-z to model-z mapping from observed anchors only.
        # Held-out tensors are not opened until downstream evaluation.
        train_model_z = []
        for slide_name in u_name_list:
            anchor = torch.load(
                os.path.join(data_dir, f'shuffled_{slide_name}.pt'),
                map_location='cpu',
            )
            train_model_z.append(float(torch.mean(anchor[:, 2]).item()))
        train_model_z = np.asarray(train_model_z, dtype=float)
        model_z_differences = np.diff(train_model_z)
        if not (
                np.all(model_z_differences > 0)
                or np.all(model_z_differences < 0)):
            raise ValueError(
                'Observed training tensors must be strictly monotonic in mean z.'
            )

        target_model_z = np.interp(
            target_z_array,
            train_z_array,
            train_model_z,
        )
        required_targets = [
            (name, float(nominal_z), float(model_z))
            for name, nominal_z, model_z in zip(
                target_slide_names,
                target_z_array,
                target_model_z,
            )
        ]
        target_output_rows = []

    existing_outputs = [
        name for name in os.listdir(result_dir)
        if name.endswith('_forward.npy') and name[:-12].isdigit()
    ]
    if existing_outputs:
        raise FileExistsError(
            f'fine_inference requires an empty result directory; found '
            f'{len(existing_outputs)} existing numbered output file(s) in '
            f'{result_dir}.'
        )

    output_count = trainer.fine_infer(
        data_dir,
        u_name_list,
        mode,
        defined_d,
        result_dir,
        batch_num,
        device,
        required_targets=required_targets,
        target_output_rows=target_output_rows,
    )

    mapping_path = None
    if target_output_rows is not None:
        mapped_names = {row['target_slide_name'] for row in target_output_rows}
        missing = set(target_slide_names) - mapped_names
        if missing:
            raise RuntimeError(
                f'No exact interpolation output was recorded for: {sorted(missing)}'
            )
        mapping_path = os.path.join(result_dir, 'target_depth_mapping.csv')
        pd.DataFrame(target_output_rows).sort_values(
            'target_nominal_z'
        ).to_csv(mapping_path, index=False)

    return {
        'n_output_layers': int(output_count),
        'target_depth_mapping_path': mapping_path,
        'n_requested_targets': 0 if target_output_rows is None else len(target_output_rows),
    }


def edge_inference(
        experiment_dir,
        data_dir,
        train_slide_names,
        train_z,
        target_slide_names,
        target_z,
        mode,
        defined_d,
        result_dir,
        batch_num,
        device='cuda'):
    """Run normal interpolation plus leakage-free left/right extrapolation.

    ``train_z`` and ``target_z`` are nominal acquisition depths. The model maps
    them into its tensor mean-z coordinate using training anchors only; held-out
    tensors are never loaded or used as boundary conditions.
    """
    model_path = os.path.join(experiment_dir, 'model.pt')
    config_path = os.path.join(experiment_dir, 'config.yml')
    missing = [
        path for path in (model_path, config_path)
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            f'Edge inference requires a trained model and config; missing: {missing}'
        )

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    TrainerClass = getattr(training_module, config['trainer'])
    trainer = TrainerClass(device=device, **config['params'])
    trainer.load(model_path)
    trainer.to(device)
    trainer.eval()
    print('Pretrained Model Loaded!')

    os.makedirs(result_dir, exist_ok=True)
    existing_outputs = [
        name for name in os.listdir(result_dir)
        if name.endswith('_forward.npy') and name[:-12].isdigit()
    ]
    if existing_outputs:
        raise FileExistsError(
            f'Edge inference requires an empty result directory; found '
            f'{len(existing_outputs)} existing numbered output file(s) in {result_dir}.'
        )

    return trainer.edge_infer(
        data_dir=data_dir,
        train_slide_names=train_slide_names,
        train_z=train_z,
        target_slide_names=target_slide_names,
        target_z=target_z,
        mode=mode,
        defined_d=defined_d,
        result_dir=result_dir,
        batch_num=batch_num,
        device=device,
    )
