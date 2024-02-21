import os
import yaml
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.logger import get_logger
from gravit.models import build_model, get_loss_func
from gravit.datasets import GraphDataset
from gravit.utils.formatter import get_formatting_data_dict, get_formatted_preds
from gravit.utils.eval_tool import get_eval_score

# Tensor board
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Ray tunner
from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

def load_data(data_dir="./data"):
    trainset = GraphDataset(os.path.join(data_dir, 'train'))
    testset = GraphDataset(os.path.join(data_dir, 'val'))
    print(f"ROCHA trainset size {len(trainset)} data_dir = {data_dir}")
    return trainset, testset

def train(cfg):
    """
    Run the training process given the configuration
    """

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    if cfg['split'] is not None:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')
    os.makedirs(path_result, exist_ok=True)

    # Prepare the logger and save the current configuration for future reference
    logger = get_logger(path_result, file_name='train')
    logger.info(cfg['exp_name'])
    logger.info('Saving the configuration file')
    with open(os.path.join(path_result, 'cfg.yaml'), 'w') as f:
        yaml.dump({k: v for k, v in cfg.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, device)
    print(path_graphs)
    trainset = GraphDataset(os.path.join(path_graphs, 'train'))
    print(f"ROCHA trainset size {len(trainset)} data_dir = {path_graphs}")
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(GraphDataset(os.path.join(path_graphs, 'val')))

    # Prepare the experiment
    loss_func = get_loss_func(cfg)
    loss_func_val = get_loss_func(cfg, 'val')
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['sch_param'])

    # Run the training process
    logger.info('Training process started')

    min_loss_val = float('inf')
    for epoch in range(1, cfg['num_epoch']+1):
        model.train()

        # Train for a single epoch
        loss_sum = 0.
        for data in train_loader:
            optimizer.zero_grad()

            x, y = data.x.to(device), data.y.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if cfg['use_spf']:
                c = data.c.to(device)

            logits = model(x, edge_index, edge_attr, c)
            tmp = logits
            if cfg['use_ref']:
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
        loss_val, score = val(val_loader, cfg['use_spf'], model, device, loss_func_val)

        # Save the best-performing checkpoint
        if loss_val < min_loss_val:
            min_loss_val = loss_val
            epoch_best = epoch
            torch.save(model.state_dict(), os.path.join(path_result, 'ckpt_best.pt'))

        # Log the losses for every epoch
        logger.info(f'Epoch [{epoch:03d}|{cfg["num_epoch"]:03d}] loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, score: {score} best: epoch {epoch_best:03d}')
        writer.add_scalar("Loss/train", loss_train, epoch)
        writer.add_scalar("Loss/val", loss_val, epoch)

    logger.info('Training finished')
    logger.info(model)

def val(val_loader, use_spf, model, device, loss_func):
    """
    Run a single validation process
    """

    # Load the feature files to properly format the evaluation results
    data_dict = get_formatting_data_dict(cfg)

    model.eval()
    loss_sum = 0
    preds_all = []
    with torch.no_grad():
        for data in val_loader:
            g = data.g.tolist()
            x, y = data.x.to(device), data.y.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            c = None
            if use_spf:
                c = data.c.to(device)

            logits = model(x, edge_index, edge_attr, c)
            loss = loss_func(logits, y)
            loss_sum += loss.item()

            # Change the format of the model output
            preds = get_formatted_preds(cfg, logits, g, data_dict)
            preds_all.extend(preds)

    eval_score = get_eval_score(cfg, preds_all)
    return loss_sum / len(val_loader), eval_score


if __name__ == "__main__":
    args = get_args()
    cfg = get_cfg(args)

    data_dir=os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    if cfg['split'] is not None:
        data_dir = os.path.join(data_dir, f'split{cfg["split"]}')
    load_data(data_dir)

    train(cfg)

    writer.flush()
    writer.close()
