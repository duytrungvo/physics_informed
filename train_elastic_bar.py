from argparse import ArgumentParser
import yaml
import torch
from models import FNO1d, FCNet, NNet
from train_utils import Adam
from train_utils.datasets import Loader_1D
from train_utils.train_1d import train_1d
from train_utils.losses import LpLoss, FDM_ElasticBar_Order2, zeros_loss, FDM_ReducedOrder_ElasticBar
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

def run(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    dataset = Loader_1D(data_config['datapath'],
                           nx=data_config['nx'],
                           sub=data_config['sub'],
                           in_dim=data_config['in_dim'],
                           out_dim=data_config['out_dim'])

    train_loader = dataset.make_loader(n_sample=data_config['n_sample'],
                                       batch_size=config['train']['batchsize'],
                                       start=data_config['offset'])

    # define model
    model_config = config['model']
    if model_config['name'] == 'fno':
        model = FNO1d(modes=model_config['modes'],
                      fc_dim=model_config['fc_dim'],
                      layers=model_config['layers'],
                      in_dim=data_config['in_dim'],
                      out_dim=data_config['out_dim'],
                      act=model_config['act']).to(device)
        if model_config['apply_output_transform'] == 'yes' and data_config['out_dim'] == 1:
            model.apply_output_transform(
                [lambda x, y: x * y + 0.0]
            )
        if model_config['apply_output_transform'] == 'yes' and data_config['out_dim'] == 2:
            model.apply_output_transform(
                [lambda x, y: x * y + 0.0,
                 lambda x, y: (data_config['L'] - x) * y + data_config['P0']]
            )
    if model_config['name'] == 'fcn':
        model = FCNet(
            layers=np.concatenate(([data_config['in_dim']], model_config['layers'][1:], [data_config['out_dim']]))).to(
            device)
    if model_config['name'] == 'nn':
        model = NNet(layers=np.concatenate(([dataset.s], model_config['layers'][1:], [dataset.s]))).to(device)
        if model_config['apply_output_transform'] == 'yes':
            model.apply_output_transform(
                lambda x, y: x * y + 0.0
            )


    # train
    train_config = config['train']
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=train_config['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=train_config['milestones'],
                                                     gamma=train_config['scheduler_gamma'])
    if train_config['pino_loss'] == 'zero':
        pino_loss = zeros_loss
    if train_config['pino_loss'] == '2nd_order':
        pino_loss = FDM_ElasticBar_Order2
    if train_config['pino_loss'] == 'reduced_order':
        pino_loss = FDM_ReducedOrder_ElasticBar
    train_1d(model,
             train_loader,
             optimizer,
             scheduler,
             config,
             pino_loss=pino_loss,
             log=False,
             use_tqdm=True)

    path = config['train']['save_dir']
    ckpt_dir = 'checkpoints/%s/' % path
    loss_history = np.loadtxt(ckpt_dir+'train_loss_history.txt')
    plt.semilogy(range(len(loss_history)), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('$Loss$')
    plt.tight_layout()
    plt.legend(['train_loss', 'pde_mse', 'bc_l_mse', 'bc_r_mse', 'data_l2'])
    plt.show()

def test(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    dataset = Loader_1D(data_config['datapath'],
                        nx=data_config['nx'],
                        sub=data_config['sub'],
                        in_dim=data_config['in_dim'],
                        out_dim=data_config['out_dim'])

    data_loader = dataset.make_loader(n_sample=data_config['n_sample'],
                                      batch_size=config['test']['batchsize'],
                                      start=data_config['offset'])

    model_config = config['model']
    if model_config['name'] == 'fno':
        model = FNO1d(modes=model_config['modes'],
                      fc_dim=model_config['fc_dim'],
                      layers=model_config['layers'],
                      in_dim=data_config['in_dim'],
                      out_dim=data_config['out_dim'],
                      act=model_config['act']).to(device)
        if model_config['apply_output_transform'] == 'yes' and data_config['out_dim'] == 1:
            model.apply_output_transform(
                [lambda x, y: x * y + 0.0]
            )
        if model_config['apply_output_transform'] == 'yes' and data_config['out_dim'] == 2:
            model.apply_output_transform(
                [lambda x, y: x * y + 0.0,
                 lambda x, y: (data_config['L'] - x) * y + data_config['P0']]
            )
    if model_config['name'] == 'fcn':
        model = FCNet(
            layers=np.concatenate(([data_config['in_dim']], model_config['layers'][1:], [data_config['out_dim']]))).to(
            device)
    if model_config['name'] == 'nn':
        model = NNet(layers=np.concatenate(([dataset.s], model_config['layers'][1:], [dataset.s]))).to(device)
        if model_config['apply_output_transform'] == 'yes':
            model.apply_output_transform(
                lambda x, y: x * y + 0.0
            )


    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)


    myloss = LpLoss(size_average=True)
    model.eval()
    s = int(np.ceil(data_config['nx'] / data_config['sub']))
    test_x = np.zeros((data_config['n_sample'], s, data_config['in_dim']))
    preds_y = np.zeros((data_config['n_sample'], s, data_config['out_dim']))
    test_y = np.zeros((data_config['n_sample'], s, data_config['out_dim']))
    test_err = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data_x, data_y = data
            data_x, data_y = data_x.to(device), data_y.to(device)
            pred_y = model(data_x).reshape(data_y.shape)
            data_loss = myloss(pred_y, data_y)
            test_err.append(data_loss.item())
            test_x[i] = data_x.cpu().numpy()
            test_y[i] = data_y.cpu().numpy()
            preds_y[i] = pred_y.cpu().numpy()

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))
    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==')

    non_dim = data_config['E'] * data_config['A0'] / (data_config['P0'] * data_config['L'])
    for i in range(3):
        key = np.random.randint(0, data_config['n_sample'])
        x_plot = test_x[key]
        y_true_plot = test_y[key]
        y_pred_plot = preds_y[key]

        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_plot[:, 1], x_plot[:, 0]/data_config['A0'])
        plt.xlabel('$x$')
        plt.ylabel('$A/A_0$')
        plt.title(f'Input $A(x)$')
        plt.xlim([0, 1])
        plt.ylim([0, np.max(test_x[:, :, 0])])

        plt.subplot(1, 2, 2)
        plt.plot(x_plot[:, 1], y_pred_plot*non_dim, 'r-x', label='predict sol')
        plt.plot(x_plot[:, 1], y_true_plot*non_dim, 'b-.o', mfc='none', label='exact sol')
        plt.xlabel('$x$')
        plt.ylabel(r'$u$')
        # plt.ylim([0, 1])
        plt.legend()
        plt.title(f'Predict and exact $u(x)$')
        plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, help='train or test')
    args = parser.parse_args()

    # read data
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(config)
    else:
        test(config)

    print('Done')