import os
import yaml
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.logger import get_logger
from gravit.models import build_model, get_loss_func
from gravit.datasets import GraphDataset

# Ray tunner
import ray
from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from functools import partial

def load_data(data_dir="./data"):
    trainset = GraphDataset(os.path.join(data_dir, 'train'))
    testset = GraphDataset(os.path.join(data_dir, 'val'))
    print(f"ROCHA trainset size {len(trainset)} data_dir = {data_dir}")
    return trainset, testset

def train(config, data_dir):
    """
    Run the training process given the configuration
    """

    # Input and output paths
    path_result = os.path.join(config['root_result'], f'{config["exp_name"]}')
    os.makedirs(path_result, exist_ok=True)

    # Prepare the logger and save the current configuration for future reference
    logger = get_logger(path_result, file_name='train')
    logger.info(config['exp_name'])
    logger.info('Saving the configuration file')
    with open(os.path.join(path_result, 'config.yaml'), 'w') as f:
        yaml.dump({k: v for k, v in config.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    # Build a model and prepare the data loaders
    logger.info('Preparing a model')
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = build_model(config, device)
    print(data_dir)
    logger.info('Preparing data loaders')
    trainset, testset = load_data(data_dir)
    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(testset)

    logger.info('Looking at Checkpoint')
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
        
    # Prepare the experiment
    loss_func = get_loss_func(config)
    loss_func_val = get_loss_func(config, 'val')
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['sch_param'])

    # Run the training process
    logger.info('Training process started')

    min_loss_val = float('inf')
    for epoch in range(start_epoch, config['num_epoch']+1):
        model.train()

        # Train for a single epoch
        loss_sum = 0.
        for data in train_loader:
            optimizer.zero_grad()

            x, y = data.x.to(device), data.y.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if config['use_spf']:
                c = data.c.to(device)

            logits = model(x, edge_index, edge_attr, c)
            tmp = logits
            if config['use_ref']:
                tmp = logits[-1]

            tmp = torch.softmax(tmp.detach().cpu(), dim=1).max(dim=1)[1].tolist()
            loss = loss_func(logits, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

        # Adjust the learning rate
        scheduler.step()

        loss_train = loss_sum / len(train_loader)

        # Get the validation loss
        loss_val, val_steps = val(val_loader, config['use_spf'], model, device, loss_func_val)

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": loss_val / val_steps},
            checkpoint=checkpoint,
        )

        # Save the best-performing checkpoint
        if loss_val < min_loss_val:
            min_loss_val = loss_val
            epoch_best = epoch
            torch.save(model.state_dict(), os.path.join(path_result, 'ckpt_best.pt'))

        # Log the losses for every epoch
        logger.info(f'Epoch [{epoch:03d}|{config["num_epoch"]:03d}] loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, best: epoch {epoch_best:03d}')

    logger.info('Training finished')


def val(val_loader, use_spf, model, device, loss_func):
    """
    Run a single validation process
    """

    model.eval()
    loss_sum = 0
    val_steps = 0
    with torch.no_grad():
        for data in val_loader:
            x, y = data.x.to(device), data.y.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if use_spf:
                c = data.c.to(device)

            logits = model(x, edge_index, edge_attr, c)
            loss = loss_func(logits, y)
            loss_sum += loss.item()
            val_steps += 1

    return loss_sum / len(val_loader), val_steps


if __name__ == "__main__":
    args = get_args()
    config = get_cfg(args)

    data_dir=os.path.join(config['root_data'], f'graphs/{config["graph_name"]}')
    if config['split'] is not None:
        data_dir = os.path.join(data_dir, f'split{config["split"]}')

    ray.init()
    print("ray.nodes()")
    print(ray.nodes())
    print("ray.cluster_resources()")
    print(ray.cluster_resources())
    print("ray.available_resources()")
    print(ray.available_resources())

    load_data(data_dir)

    config["channel1"] = tune.choice([2**i for i in range(5,10)])
    config["channel2"] = tune.choice([2**i for i in range(5,10)])
    config["lr"] = tune.loguniform(1e-6, 1e-2)
    config["wd"] = tune.loguniform(1e-7, 1e-3),

    print(f"Data dir {data_dir}")
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=config['num_epoch'],
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=config,
        num_samples=10,
        scheduler=scheduler,
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

