import os
import yaml
import torch
import argparse
import h5py
#from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
#from gravit.models import build_model
#from gravit.datasets import GraphDataset
#from gravit.utils.formatter import get_formatting_data_dict, get_formatted_preds
from gravit.utils.eval_tool import get_eval_score

# generate random integer values
from random import seed
from random import randint, uniform
# seed random number generator


def gen_random_preds(cfg):
    """
    Generates random predictions for a randomly selected ~20% of videos 
    """

    # seed(1)
    preds = []
    path_dataset = os.path.join(cfg['root_data'], f'annotations/{cfg["dataset"]}/eccv16_dataset_{cfg["dataset"].lower()}_google_pool5.h5')
    with h5py.File(path_dataset, 'r') as hdf:
        
        # Randomly select 20% of videos
        video_names = list(hdf.keys())
        rand_selected_vids = []
        amount_to_select = len(video_names)//5
        for _ in range(amount_to_select): 
            video = randint(0, len(video_names) - 1) 
            rand_selected_vids.append(video_names.pop(video))
        
        # Randomly score video samples
        for video in rand_selected_vids:
            n_steps = hdf.get(video + '/n_steps')[()]
            scores = []
            samples = []
            for sample in range(n_steps):
                scores.append(uniform(0, 1))
                samples.append(sample)
            preds.append([video, samples, scores])

        # print(f"Selected video: {preds}")

    return preds

def evaluate(cfg):
    """
    Run the evaluation process given the configuration
    """

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    if 'split' in cfg:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')

    # Prepare the logger
    logger = get_logger(path_result, file_name='eval')
    logger.info(cfg['exp_name'])

    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = build_model(cfg, device)
    # val_loader = DataLoader(GraphDataset(os.path.join(path_graphs, 'val')))
    # num_val_graphs = len(val_loader)

    # Load the trained model
    # logger.info('Loading the trained model')
    # state_dict = torch.load(os.path.join(path_result, 'ckpt_best.pt'), map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # model.eval()

    # Load the feature files to properly format the evaluation results
    logger.info('Retrieving the formatting dictionary')
    # data_dict = get_formatting_data_dict(cfg)

    # Run the evaluation process
    logger.info('Evaluation process started')

    preds_all = []
    # with torch.no_grad():
    #    for i, data in enumerate(val_loader, 1):
    #        g = data.g.tolist()
    #        x = data.x.to(device)
    #        edge_index = data.edge_index.to(device)
    #        edge_attr = data.edge_attr.to(device)
    #        c = None
    #        if cfg['use_spf']:
    #            c = data.c.to(device)

    #        logits = model(x, edge_index, edge_attr, c)

            # Change the format of the model output
    #        preds = get_formatted_preds(cfg, logits, g, data_dict)
    #        preds_all.extend(preds)

    #        logger.info(f'[{i:04d}|{num_val_graphs:04d}] processed')
    preds_all.extend(gen_random_preds(cfg))
    # Compute the evaluation score
    logger.info('Computing the evaluation score')
    # print(f"{cfg} and {preds_all}")
    eval_score = get_eval_score(cfg, preds_all)
    logger.info(f'{cfg["eval_type"]} evaluation finished: {eval_score}')


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
        raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

    args.cfg = os.path.join(path_result, 'cfg.yaml')
    cfg = get_cfg(args)

    evaluate(cfg)
