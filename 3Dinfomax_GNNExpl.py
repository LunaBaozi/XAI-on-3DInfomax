import argparse
import os
import re

from icecream import install
from plot_function import my_plot_function

import yaml
from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from datasets.samplers import *  # do not remove

from datasets.qm9_dataset import QM9Dataset

from train import load_model, get_arguments

# turn on for debugging C code like Segmentation Faults
import faulthandler
faulthandler.enable()
install()

import torch
from datasets.qm9_dataset import QM9Dataset
from dgl.nn import GNNExplainer


# pretraining checkpoint provided in repo
DEFAULT_MODEL_CHECKPOINT = 'runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt'


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/pna.yml') #12.yml
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset', type=str, default='qm9', help='[qm9, zinc, drugs, geom_qm9, molhiv]')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--critic_loss', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--critic_loss_params', type=dict, default={},
                   help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--linear_probing_samples', type=int, default=500,
                   help='number of samples to use for linear probing in the run_eval_per_epoch function of the self supervised trainer')
    p.add_argument('--num_conformers', type=int, default=3,
                   help='number of conformers to use if we are using multiple conformers on the 3d side')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='mae_denormalized', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--pretrain_checkpoint', type=str, default=DEFAULT_MODEL_CHECKPOINT, help='Specify path to finetune from a pretrained checkpoint')  #default=DEFAULT_MODEL_FINETUNED,
    p.add_argument('--transfer_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--frozen_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--exclude_from_transfer', default=[],
                   help='parameters that usually should not be transferred like batchnorm params')
    p.add_argument('--transferred_lr', type=float, default=None, help='set to use a different LR for transfer layers')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--required_data', default=['dgl_graph', 'smiles'],
                   help='what will be included in a batch like [dgl_graph, targets, dgl_graph3d]')  #'targets'
    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--use_e_features', default=True, type=bool, help='ignore edge features if set to False')
    p.add_argument('--targets', default=[], help='properties that should be predicted')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--dist_embedding', type=bool, default=False, help='add dist embedding to complete graphs edges')
    p.add_argument('--num_radial', type=int, default=6, help='number of frequencies for distance embedding')
    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='PNA', help='Classname of one of the models in the models dir')  #MPNN #'PNA'
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--model3d_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--model3d_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--critic_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--critic_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--trainer', type=str, default='contrastive', help='[contrastive, byol, alternating, philosophy]')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--force_random_split', type=bool, default=False, help='use random split for ogb')
    p.add_argument('--reuse_pre_train_data', type=bool, default=False, help='use all data instead of ignoring that used during pre-training')
    p.add_argument('--transfer_3d', type=bool, default=False, help='set true to load the 3d network instead of the 2d network')
    p.add_argument('-f')
    
    args, _ = p.parse_known_args()
    return  args   #p.parse_args()


def load_model(args, data, device):
    print('Model type: ', args.model_type)
    model = globals()[args.model_type](avg_d=data.avg_degree if hasattr(data, 'avg_degree') else 1, device=device,
                                       **args.model_parameters)
    print('Pre-train checkpoint: ', args.pretrain_checkpoint)
    if args.pretrain_checkpoint:
        print("READING CHECKPOINT")
        # get arguments used during pretraining
        with open(os.path.join(os.path.dirname(args.pretrain_checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            pretrain_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        pretrain_args = argparse.Namespace()
        pretrain_args.__dict__.update(pretrain_dict)
        checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
        # get all the weights that have something from 'args.transfer_layers' in their keys name
        # but only if they do not contain 'teacher' and remove 'student.' which we need for loading from BYOLWrapper
        weights_key = 'model3d_state_dict' if args.transfer_3d == True else 'model_state_dict'
        pretrained_gnn_dict = {re.sub('^gnn\.|^gnn2\.', 'node_gnn.', k.replace('student.', '')): v
                               for k, v in checkpoint[weights_key].items() if any(
                transfer_layer in k for transfer_layer in args.transfer_layers) and 'teacher' not in k and not any(
                to_exclude in k for to_exclude in args.exclude_from_transfer)}
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_gnn_dict)  # update the gnn layers with the pretrained weights
        model.load_state_dict(model_state_dict)
        if args.reuse_pre_train_data:
            return model, 0, pretrain_args.dataset == args.dataset
        else:
            return model, pretrain_args.num_train, pretrain_args.dataset == args.dataset
    return model, None, False


def get_arguments():
    args = parse_arguments()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}

    if args.checkpoint:  # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

    return args



args = get_arguments()
# device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
device = torch.device('cpu')

dataset = QM9Dataset(return_types=args.required_data, target_tasks=args.targets, device=device,
                          dist_embedding=args.dist_embedding, num_radial=args.num_radial)
model, num_pretrain, transfer_from_same_dataset = load_model(args, data=dataset, device=device)
# eval_qm9(args, device, metrics_dict, model, num_pretrain, transfer_from_same_dataset, dataset)

qm9_processed = torch.load('dataset/QM9/processed/qm9_processed.pt')



def XAI(graph_idx, model, dataset):

    start_atom = qm9_processed['atom_slices'][graph_idx]
    end_atom = qm9_processed['atom_slices'][graph_idx+1]
    start_edge = qm9_processed['edge_slices'][graph_idx]
    end_edge = qm9_processed['edge_slices'][graph_idx+1]
    # source_edges = qm9_processed['edge_indices'][0][start_edge:end_edge]
    # target_edges = qm9_processed['edge_indices'][1][start_edge:end_edge]
    atom_features = qm9_processed['atom_features'][start_atom:end_atom]
    # edge_features = qm9_processed['edge_features'][start_edge:end_edge]

    g = dataset[graph_idx]
    graph = g[0]
    smiles = g[1]
    print((type(smiles), smiles))

    model = model.to(device)
    graph = graph.to(device)
    atom_features = atom_features.to(device)

    # DGL GNNExplainer
    explainer = GNNExplainer(model, num_hops=3)

    mol_dir = f'./explanations/{graph_idx}_{smiles}/'
    if not os.path.exists(mol_dir):
        os.makedirs(mol_dir)

    for i in range(3):
        feat_mask, edge_mask = explainer.explain_graph(graph, atom_features) #, edge_features atom_features
        print('Atom feature mask: ', feat_mask)
        print('Edge feature mask: ', edge_mask)        
        my_plot_function(mol_dir, graph_idx, i, smiles, edge_mask)
    
    return


# for i in range(124440, 124450):
#     XAI(i, model=model, dataset=dataset)
XAI(124440, model=model, dataset=dataset)