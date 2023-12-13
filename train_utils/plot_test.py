from tqdm import tqdm
import numpy as np

import torch

from .losses import LpLoss, darcy_loss, PINO_loss

try:
    import wandb
except ImportError:
    wandb = None

from train_utils.datasets import BurgersLoader
from models import FNO2d
import matplotlib.pyplot as plt

def plot_pred(data_config, test_x, test_y, preds_y, error_index):
    for i in range(3):
        # key = np.random.randint(0, data_config['n_sample'])
        key = error_index[-1-i]
        x_plot = test_x[key]
        y_true_plot = test_y[key]
        y_pred_plot = preds_y[key]

        fig = plt.figure(figsize=(10, 12))
        if data_config['out_dim'] == 1:
            plt.subplot(3, 1, 1)
            plt.plot(x_plot[:, -1], x_plot[:, 0] / data_config['I0'])
            plt.xlabel('$x$')
            plt.ylabel('$I/I_0$')
            plt.title(f'Input $I(x)$')
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            plt.subplot(3, 1, 2)
            plt.plot(x_plot[:, -1], x_plot[:, 1] / data_config['q0'])
            plt.xlabel('$x$')
            plt.ylabel('$q/q_0$')
            plt.title(f'Input $q(x)$')
            plt.xlim([0, 1])
            plt.ylim([-0.1, 1.1])

            plt.subplot(3, 1, 3)
            plt.plot(x_plot[:, -1], y_pred_plot[:, 0], 'r', label='predict sol')
            plt.plot(x_plot[:, -1], y_true_plot[:, 0], 'b', label='exact sol')
            plt.xlabel('$x$')
            plt.ylabel(r'$w$')
            # plt.ylim([0, 1])
            plt.legend()
            plt.grid(visible=True)
            plt.title(f'Predict and exact $w(x)$')
            plt.tight_layout()
        if data_config['out_dim'] == 2:
           plt.subplot(4, 1, 1)
           plt.plot(x_plot[:, -1], x_plot[:, 0] / data_config['I0'])
           plt.xlabel('$x$')
           plt.ylabel('$I/I_0$')
           plt.title(f'Input $I(x)$')
           plt.xlim([0, 1])
           # plt.ylim([0, 2])

           plt.subplot(4, 1, 2)
           plt.plot(x_plot[:, -1], x_plot[:, 1] / data_config['q0'])
           plt.xlabel('$x$')
           plt.ylabel('$q/q_0$')
           plt.title(f'Input $q(x)$')
           plt.xlim([0, 1])
           plt.ylim([-0.1, 1.1])

           plt.subplot(4, 1, 3)
           plt.plot(x_plot[:, -1], y_pred_plot[:, 0], 'r', label='predict sol')
           plt.plot(x_plot[:, -1], y_true_plot[:, 0], 'b', label='exact sol')
           plt.xlabel('$x$')
           plt.ylabel(r'$w$')
           # plt.ylim([0, 1])
           plt.legend()
           plt.grid(visible=True)
           plt.title(f'Predict and exact $w(x)$')

           plt.subplot(4, 1, 4)
           plt.plot(x_plot[:, -1], y_pred_plot[:, 1], 'r', label='predict sol')
           plt.plot(x_plot[:, -1], y_true_plot[:, 1], 'b', label='exact sol')
           plt.xlabel('$x$')
           plt.ylabel(r'$M$')
           # plt.ylim([0, 1])
           plt.legend()
           plt.grid(visible=True)
           plt.title(f'Predict and exact $M(x)$')
           plt.tight_layout()

        plt.show()

def plot_predictions(key, test_x, test_y, preds_y, print_index=False, save_path=None, font_size=None):
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})
    pred = preds_y[key]
    true = test_y[key]

    a = test_x[key]
    Nt, Nx, _ = a.shape
    u0 = a[0, :, 0]
    T = a[:, :, 2]
    X = a[:, :, 1]
    x = X[0]

    # Plot
    fig = plt.figure(figsize=(23, 5))
    plt.subplot(1, 4, 1)

    plt.plot(x, u0)
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.title('Intial Condition $u(x)$')
    plt.xlim([0, 1])
    # plt.tight_layout()

    plt.subplot(1, 4, 2)
    # plt.pcolor(XX,TT, S_test, cmap='jet')
    plt.pcolormesh(X, T, true, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title(f'Exact $u(x,t)$')
    # plt.tight_layout()
    plt.axis('square')

    plt.subplot(1, 4, 3)
    # plt.pcolor(XX,TT, S_pred, cmap='jet')
    plt.pcolormesh(X, T, pred, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title(f'Predict $u(x,t)$')
    plt.axis('square')

    # plt.tight_layout()

    plt.subplot(1, 4, 4)
    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    plt.pcolormesh(X, T, pred - true, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Absolute Error')
    plt.axis('square')

    if save_path is not None:
        plt.savefig(f'{save_path}.png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_solution(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = BurgersLoader(data_config['datapath'],
                            nx=data_config['nx'], nt=data_config['nt'],
                            sub=data_config['sub'], sub_t=data_config['sub_t'], new=True)
    dataloader = dataset.make_loader(n_sample=data_config['n_sample'],
                                     batch_size=config['test']['batchsize'],
                                     start=data_config['offset'])

    model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  act=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])

    Nx = config['data']['nx'] // config['data']['sub']
    Nt = config['data']['nt'] // config['data']['sub_t'] + 1
    Ntest = config['data']['n_sample']
    model.eval()
    test_x = np.zeros((Ntest,Nt,Nx,3))
    preds_y = np.zeros((Ntest,Nt,Nx))
    test_y = np.zeros((Ntest,Nt,Nx))
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data_x, data_y = data
            data_x, data_y = data_x.to(device), data_y.to(device)
            pred_y = model(data_x).reshape(data_y.shape)
            test_x[i] = data_x.cpu().numpy()
            test_y[i] = data_y.cpu().numpy()
            preds_y[i] = pred_y.cpu().numpy()

    for key in range(3):
        plot_predictions(key, test_x, test_y, preds_y, print_index=False, save_path=None, font_size=11)
