import os
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



def train(trainer, experiment_dir, dataset, batch_num, epochs_list, checkpoint_every, backup_every):
    checkpoint_count = 0
    tqdm.write('Dataset loaded!')
    if len(epochs_list) == 4:
        tqdm.write('---------4 stage training start----------')
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[0], 'prior', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[1], 'coo', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[2], 'seq', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[3], 'joint', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.save(os.path.join(experiment_dir, 'model.pt'))
    elif len(epochs_list) == 3:
        tqdm.write('---------3 stage training start----------')
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[0], 'coo', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[1], 'seq', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[2], 'joint', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.save(os.path.join(experiment_dir, 'model.pt'))
    elif len(epochs_list) == 2:
        tqdm.write('---------2 stage training start----------')
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[0], 'coo', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[1], 'seq', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.iterations = 0
        trainer.save(os.path.join(experiment_dir, 'model.pt'))
    else:
        trainer.iterations = 0
        train_stage(trainer, experiment_dir, dataset, batch_num, epochs_list[0], 'joint', checkpoint_count,
                    checkpoint_every, backup_every)
        trainer.save(os.path.join(experiment_dir, 'model.pt'))

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