import os
import yaml
import torch
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
from gravit.models import build_model
from gravit.datasets import GraphDataset
from gravit.utils.formatter import get_formatting_data_dict, get_formatted_preds
from torch.profiler import profile, record_function, ProfilerActivity
from gravit.utils.eval_tool import get_eval_score


def evaluate(cfg):
    """
    Run the evaluation process given the configuration
    """
    # Path to experiment results
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')

    # Input path
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')

    # Prepare the logger
    logger = get_logger(path_result, file_name='eval')
    logger.info(cfg['exp_name'])

    # Build a model
    logger.info('Preparing a model and data loaders')
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = build_model(cfg, device)

    # Init model with dummy inputs
    x_dummy = torch.tensor(np.array(np.random.rand(10, 1024), dtype=np.float32), dtype=torch.float32).to(device)
    node_source_dummy = np.random.randint(10, size=5)
    node_target_dummy = np.random.randint(10, size=5)
    edge_index_dummy = torch.tensor(np.array([node_source_dummy, node_target_dummy], dtype=np.int64), dtype=torch.long).to(device)
    signs = np.sign(node_source_dummy - node_target_dummy)
    edge_attr_dummy = torch.tensor(signs, dtype=torch.float32).to(device)
    model(x_dummy, edge_index_dummy, edge_attr_dummy, None)

    # Load the trained model
    logger.info('Loading the trained model')
    state_dict = torch.load(os.path.join(path_result, 'ckpt_best.pt'), map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()

    # Load the feature files to properly format the evaluation results
    logger.info('Retrieving the formatting dictionary')
    data_dict = get_formatting_data_dict(cfg)


    splits = 1
    if 'split' in cfg:
        splits = 5
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   profile_memory=True,
                   with_flops=True,
                   record_shapes=True,
                   with_stack=True)
    all_scores = []
    all_taus = []
    all_rhos = []
    for split in range(1, splits + 1):

        if splits > 1:
            path_input = os.path.join(path_graphs, f'split{split}')

        # Prepare the data loader
        val_loader = DataLoader(GraphDataset(os.path.join(path_input, 'val')))
        num_val_graphs = len(val_loader)

        # Run the evaluation process
        logger.info(f'Evaluation process started for split{split}')

        preds_all = []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 1):
                g = data.g.tolist()
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device)
                c = None
                if cfg['use_spf']:
                    c = data.c.to(device)
                prof.start()
                with record_function("model_inference"):
                    logits = model(x, edge_index, edge_attr, c)
                prof.stop()
                # Change the format of the model output
                preds = get_formatted_preds(cfg, logits, g, data_dict)
                preds_all.extend(preds)

                logger.info(f'Split{split} - [{i:04d}|{num_val_graphs:04d}] processed')

        # Compute the evaluation score
        logger.info('Computing the evaluation score')
        eval_score, tau, rho = get_eval_score(cfg, preds_all)
        all_scores.append(eval_score)
        all_taus.append(tau)
        all_rhos.append(rho)
        logger.info(f'{cfg["eval_type"]} evaluation finished on split{split}: F-1 = {eval_score}, Tau = {tau}, Rho = {rho}')

    logger.info(f'Final average evaluation score: F-1 = {np.mean(all_scores)}, Tau = {np.mean(all_taus)}, Rho = {np.mean(all_rhos)}')
    prof.export_memory_timeline("Memory-Gravi-T.html", "cpu")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

if __name__ == "__main__":
    """ 
    Evaluate the trained model from the experiment "exp_name"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--root_result',   type=str,   help='Root directory to output', default='./results')
    parser.add_argument('--dataset',       type=str,   help='Name of the dataset')
    parser.add_argument('--exp_name',      type=str,   help='Name of the experiment', required=True)
    parser.add_argument('--eval_type',     type=str,   help='Type of the evaluation', required=True)

    args = parser.parse_args()

    path_result = os.path.join(args.root_result, args.exp_name)
    if not os.path.isdir(path_result):
        raise ValueError(f'Please run the training experiment "{path_result}" first')

    args.cfg = os.path.join(path_result, 'cfg.yaml')
    cfg = get_cfg(args)

    evaluate(cfg)
