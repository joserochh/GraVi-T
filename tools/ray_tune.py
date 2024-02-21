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

# Ray tunner
import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import numpy as np

def train(cfg):
    """
    Run the training process given the configuration
    """
    # Path to experiment results
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')

    # Input path
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    train_path_graphs = path_graphs
    test_path_graph = path_graphs
    if cfg['split'] is not None:
        train_path_graphs = os.path.join(train_path_graphs, f'split{cfg["split"]}')

    os.makedirs(path_result, exist_ok=True)

    # Prepare the logger and save the current configuration for future reference
    logger = get_logger(path_result, file_name='train')
    logger.info(cfg['exp_name'])
    logger.info('Saving the configuration file')
    with open(os.path.join(path_result, 'cfg.yaml'), 'w') as f:
        yaml.dump({k: v for k, v in cfg.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    # Build a model and prepare the data loaders
    logger.info('Preparing a model')
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, device)

    logger.info('Preparing data loaders')
    trainset = GraphDataset(os.path.join(train_path_graphs, 'train'))
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True)


    start_epoch = 0
        
    # Prepare the experiment
    loss_func = get_loss_func(cfg)
    loss_func_val = get_loss_func(cfg, 'val')
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['sch_param'])

    # Run the training process
    logger.info('Training process started')

    min_loss = float('inf')
    max_avg_score = 0
    all_epochs_f1 = []
    epoch_max_score = 0
    for epoch in range(start_epoch, cfg['num_epoch']+1):
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

        splits = 1
        if cfg['split'] is not None:
            splits = 5

        loss_val = 0
        score = 0
        all_scores = []
        for split in range(1, splits + 1):

            if splits > 1:
                test_path_graph = os.path.join(path_graphs, f'split{split}')

            testset = GraphDataset(os.path.join(test_path_graph, 'val'))
            val_loader = DataLoader(testset)

            # Get the validation loss
            split_loss_val, split_score = val(val_loader, cfg['use_spf'], model, device, loss_func_val)

            if split == cfg['split']:
                score = split_score
                loss_val = split_loss_val

            all_scores.append(split_score)

        avg_score = np.mean(all_scores)

        if avg_score > max_avg_score:
            max_avg_score = avg_score
            epoch_max_score = epoch

        # Save the best-performing checkpoint
        if loss_val < min_loss:
            min_loss = loss_val
            epoch_best = epoch
            epoch_best_f1 = avg_score
            torch.save(model.state_dict(), os.path.join(path_result, 'ckpt_best.pt'))

        if epoch > 45:
            all_epochs_f1.append(avg_score)

        session.report(
            {
                "loss_val": loss_val,
                "loss_train": loss_train,
                "F1": score,
                "splits_F1": avg_score,
                "max_splits_F1": max_avg_score,
                "max_splits_F1_epoch": epoch_max_score,
                "epoch_best": epoch_best,
                "splits_F1_epoch_best": epoch_best_f1,
                "avg": np.mean(all_epochs_f1)
            }
        )

        # Log the losses for every epoch
        logger.info(f'Epoch [{epoch:03d}|{cfg["num_epoch"]:03d}] loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, best: epoch {epoch_best:03d}, F-Score: {score}, Avg F-Score for all splits: {avg_score}, F1: {all_scores[0]}, F2: {all_scores[1]}, F3: {all_scores[2]}, F4: {all_scores[3]}, F5: {all_scores[4]}')

    logger.info('Training finished')


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

    cfg['root_data'] = os.path.realpath(cfg['root_data'])

    ray.init()
    print("ray.nodes()")
    print(ray.nodes())
    print("ray.cluster_resources()")
    print(ray.cluster_resources())
    print("ray.available_resources()")
    print(ray.available_resources())

    #cfg["channel1"] = tune.choice([2**i for i in range(5,10)])
    #cfg["channel2"] = tune.choice([2**i for i in range(5,10)])
    #cfg["lr"] = tune.loguniform(1e-4, 1e-2)
    #cfg["wd"] = tune.loguniform(1e-7, 1e-4)

    cfg['graph_name'] = tune.grid_search(["SumMe_10_0"])
    cfg['split'] = tune.grid_search([3])
    cfg["channel1"] = tune.grid_search([128])
    cfg["channel2"] = tune.grid_search([64])
    cfg["lr"] = tune.grid_search([1.38e-3])
    cfg["wd"] = tune.grid_search([3.30e-7])
    cfg['num_epoch'] = 150
    cfg['eval_type'] = "VS_max"

    #cfg['graph_name'] = tune.grid_search(["TVSum_5_0"])
    #cfg['split'] = tune.grid_search([2])
    #cfg["channel1"] = tune.grid_search([256])
    #cfg["channel2"] = tune.grid_search([128])
    #cfg["lr"] = tune.grid_search([1.82e-3])
    #cfg["wd"] = tune.grid_search([4.18e-5])

    scheduler = ASHAScheduler(
        metric="F1",
        mode="max",
        max_t=cfg['num_epoch'],
        grace_period=40,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train),
        resources_per_trial={"cpu": 2, "gpu": 0},
        config=cfg,
        num_samples=20,
        max_concurrent_trials=8
        #scheduler=scheduler,
    )
    
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final accuracy: {best_trial.last_result['accuracy']}")

