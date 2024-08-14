import torch
from torch.utils.data import DataLoader
from models import FNO2d, FNO2dv2
import yaml
from argparse import ArgumentParser
from train_utils.datasets import Loader2D
from train_utils import Adam
from train_utils.train_2d import train_2d_planar
import matplotlib.pyplot as plt
import numpy as np
from train_utils.eval_2d import eval_2Dplanar

torch.manual_seed(0)
np.random.seed(0)

def train_2d(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    dataset = Loader2D(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])

    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=True)
    model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=config['model']['in_dim'],
                  out_dim=config['model']['out_dim'],
                  act=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])

    train_2d_planar(model,
                      train_loader,
                      optimizer, scheduler,
                      config)

    path = config['train']['save_dir']
    ckpt_dir = 'checkpoints/%s/' % path
    loss_history = np.loadtxt(ckpt_dir + 'train_loss_history.txt')
    plt.semilogy(range(len(loss_history)), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('$Loss$')
    plt.tight_layout()
    plt.legend(['train_loss', 'pde_l2', 'data_l2'])
    plt.show()

def train_2dv1(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    dataset = Loader2D(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])

    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=True)
    model = FNO2dv2(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=config['model']['in_dim'],
                  out_dim=config['model']['out_dim'],
                  act=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
    #                  lr=config['train']['base_lr'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=config['train']['milestones'],
    #                                                  gamma=config['train']['scheduler_gamma'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['base_lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)

    train_2d_planar(model,
                      train_loader,
                      optimizer, scheduler,
                      config, use_tqdm=False)

    path = config['train']['save_dir']
    ckpt_dir = 'checkpoints/%s/' % path
    loss_history = np.loadtxt(ckpt_dir + 'train_loss_history.txt')
    plt.semilogy(range(len(loss_history)), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('$Loss$')
    plt.tight_layout()
    plt.legend(['train_loss', 'pde_l2', 'data_l2'])
    plt.show()

def test_2d(args, config):
    device = 0 if torch.cuda.is_available() else 'cpu'
    data_config = config['data']
    dataset = Loader2D(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])

    dataloader = DataLoader(dataset, batch_size=config['test']['batchsize'], shuffle=False)

    model = FNO2dv2(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=config['model']['in_dim'],
                  out_dim=config['model']['out_dim'],
                  act=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    eval_2Dplanar(model, dataloader, config, device, use_tqdm=False)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, help='train or test')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if args.mode == 'train':
        train_2dv1(args, config)
    else:
        test_2d(args, config)
