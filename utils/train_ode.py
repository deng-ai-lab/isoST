import os
import csv
import yaml
from tqdm import tqdm
from model.utils.ReadData import get_dataset,  get_group_dataset
import model as training_module
import sys


def train_stage(trainer, experiment_dir, dataset, batch_num, epochs, flag, checkpoint_count, checkpoint_every, backup_every):
    if flag == 'seq':
        trainer.freeze_coo_parameters()
        trainer.unfreeze_seq_parameters()
    elif flag == 'coo':
        trainer.unfreeze_coo_parameters()
        trainer.freeze_seq_parameters()
    else:
        trainer.unfreeze_coo_parameters()
        trainer.unfreeze_seq_parameters()
    pbar = tqdm(total=epochs, file=sys.stdout)
    pbar.write(f'\n----------{flag} training---------')
    for epoch in range(epochs):
        pbar.update(1)
        for k in range(batch_num):
            trainer.train_step(dataset=dataset[k], flag=flag)

        if (epoch + 1) % checkpoint_every == 0:
            trainer.save(os.path.join(experiment_dir, 'checkpoint_%d.pt' % checkpoint_count))
            checkpoint_count += 1

        if (epoch + 1) % backup_every == 0:
            pbar.write('\n--------- back up ----------')
            pbar.write(f'beta: %f' % trainer.loss_items[f'beta'][-1])
            pbar.write(f'loss_{flag}: %f' % trainer.loss_items[f'total loss'][-1])

    pbar.close()
    return checkpoint_count


def save_loss_artifacts(trainer, experiment_dir, stage_boundaries=None):
    """Persist the in-memory loss history as both a table and a plot."""
    loss_items = getattr(trainer, 'loss_items', {})
    if not loss_items:
        return

    keys = sorted(loss_items)
    row_count = max((len(loss_items[key]) for key in keys), default=0)
    if row_count == 0:
        return

    csv_path = os.path.join(experiment_dir, 'loss_history.csv')
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', *keys])
        for index in range(row_count):
            writer.writerow([
                index,
                *[
                    loss_items[key][index] if index < len(loss_items[key]) else ''
                    for key in keys
                ],
            ])

    total_loss = loss_items.get('total loss', [])
    if total_loss:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(len(total_loss)), total_loss, linewidth=1.2)
        for boundary in stage_boundaries or []:
            ax.axvline(boundary, color='tab:gray', linestyle='--', alpha=0.55)
        ax.set_xlabel('optimization step')
        ax.set_ylabel('total loss')
        ax.set_title('Training loss')
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(experiment_dir, 'loss_curve.png'), dpi=200, bbox_inches='tight')
        fig.savefig(os.path.join(experiment_dir, 'loss_curve.pdf'), bbox_inches='tight')
        plt.close(fig)



def train(trainer, experiment_dir, dataset, batch_num, epochs_list, checkpoint_every, backup_every):
    checkpoint_count = 0
    tqdm.write('Dataset loaded!')
    stage_names = {
        4: ['prior', 'coo', 'seq', 'joint'],
        3: ['coo', 'seq', 'joint'],
        2: ['coo', 'seq'],
    }.get(len(epochs_list), ['joint'])
    stage_epochs = epochs_list if len(epochs_list) in {2, 3, 4} else epochs_list[:1]
    stage_boundaries = []

    try:
        tqdm.write(f'---------{len(stage_names)} stage training start----------')
        completed_steps = 0
        for stage_name, stage_epoch_count in zip(stage_names, stage_epochs):
            trainer.iterations = 0
            checkpoint_count = train_stage(
                trainer,
                experiment_dir,
                dataset,
                batch_num,
                stage_epoch_count,
                stage_name,
                checkpoint_count,
                checkpoint_every,
                backup_every,
            )
            completed_steps += stage_epoch_count * batch_num
            stage_boundaries.append(completed_steps)

        trainer.save(os.path.join(experiment_dir, 'model.pt'))
    finally:
        save_loss_artifacts(trainer, experiment_dir, stage_boundaries[:-1])

def biaxial_train(
        experiment_dir,
        data_dir,
        slide_names,
        batch_num,
        config_file,
        device,
        checkpoint_every,
        backup_every,
        epoch,
        mode,
) -> object:

    # Load the configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Copy it to the experiment folder
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)

    # Instantiating the trainer according to the specified configuration
    TrainerClass = getattr(training_module, config['trainer'])
    trainer = TrainerClass(device=device, **config['params'])
    trainer.to(device)

    ###########
    # Dataset #
    ###########
    # Loading the dataset
    train_set = get_dataset(data_dir, slide_names, batch_num, device, mode)
    tqdm.write('========== Optimization ============')
    train(trainer, experiment_dir, train_set, batch_num, epoch, checkpoint_every, backup_every)



def biaxial_multi_group_train(
        experiment_dir,
        data_dir,
        grouped_slide_names,
        batch_num,
        config_file,
        device,
        checkpoint_every,
        backup_every,
        epoch,
        mode,
) -> object:

    # Load the configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Copy it to the experiment folder
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)

    # Instantiating the trainer according to the specified configuration
    TrainerClass = getattr(training_module, config['trainer'])
    trainer = TrainerClass(device=device, **config['params'])
    trainer.to(device)

    ###########
    # Dataset #
    ###########
    # Loading the dataset
    train_set = get_group_dataset(data_dir, grouped_slide_names, batch_num, device, mode)
    tqdm.write('========== Optimization ============')
    train(trainer, experiment_dir, train_set, batch_num, epoch, checkpoint_every, backup_every)
