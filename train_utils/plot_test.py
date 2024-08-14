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

def plot_pred(data_config, test_x, test_y, pred_y, error_index):
    for i in range(3):
        # key = np.random.randint(0, data_config['n_sample'])
        key = error_index[-1-i]
        x_plot = test_x[key]
        y_true_plot = test_y[key]
        y_pred_plot = pred_y[key]

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
def plot_darcy(mesh, test_x, test_y, pred_y):
    a = test_x[..., 0].squeeze()
    # a = a.squeeze()
    u_true = test_y.squeeze()
    u_pred = pred_y.squeeze()
    x = mesh[..., 0]
    y = mesh[..., 1]
    # Plot
    fig = plt.figure(figsize=(23, 5))
    plt.subplot(1, 4, 1)
    plt.pcolormesh(x, y, a, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(f'Input $a(x,y)$')
    # plt.tight_layout()
    plt.axis('square')

    plt.subplot(1, 4, 2)
    plt.pcolormesh(x, y, u_true, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(f'Exact $u(x,y)$')
    # plt.tight_layout()
    plt.axis('square')

    plt.subplot(1, 4, 3)
    plt.pcolormesh(x, y, u_pred, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(f'Predicted $u(x,y)$')
    plt.axis('square')

    plt.subplot(1, 4, 4)
    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    plt.pcolormesh(x, y, torch.abs(u_pred - u_true), cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Absolute Error')
    plt.tight_layout()
    plt.axis('square')

    plt.show()


def plot_2Dplanar(mesh, test_x, test_y, pred_y):
    a1 = test_x[..., 0].squeeze()
    a2 = test_x[..., 1].squeeze()

    u1_true = test_y[..., 0].squeeze()
    u1_pred = pred_y[..., 0].squeeze()

    u2_true = test_y[..., 1].squeeze()
    u2_pred = pred_y[..., 1].squeeze()

    u3_true = test_y[..., 2].squeeze()
    u3_pred = pred_y[..., 2].squeeze()

    u4_true = test_y[..., 3].squeeze()
    u4_pred = pred_y[..., 3].squeeze()

    u5_true = test_y[..., 4].squeeze()
    u5_pred = pred_y[..., 4].squeeze()

    x = mesh[..., 0]
    y = mesh[..., 1]
    # Plot
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_sol_2d(x, y, a1, f'Input $f_x(x,y)$')
    plt.axis('square')

    plt.subplot(1, 2, 2)
    plot_sol_2d(x, y, a2, f'Input $f_y(x,y)$')
    plt.tight_layout()
    plt.axis('square')
    plt.show()

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plot_sol_2d(x, y, u1_true, f'Exact $u(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 2)
    plot_sol_2d(x, y, u1_pred, f'Predicted $u(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 3)
    plot_abs_error(x, y, u1_pred, u1_true)
    plt.tight_layout()
    plt.axis('square')
    plt.show()

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plot_sol_2d(x, y, u2_true, f'Exact $v(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 2)
    plot_sol_2d(x, y, u2_pred, f'Predicted $v(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 3)
    plot_abs_error(x, y, u2_pred, u2_true)
    plt.tight_layout()
    plt.axis('square')
    plt.show()

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plot_sol_2d(x, y, u3_true, f'Exact $\sigma_x(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 2)
    plot_sol_2d(x, y, u3_pred, f'Predicted $\sigma_x(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 3)
    plot_abs_error(x, y, u3_pred, u3_true)
    plt.tight_layout()
    plt.axis('square')
    plt.show()

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plot_sol_2d(x, y, u4_true, f'Exact $\sigma_y(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 2)
    plot_sol_2d(x, y, u4_pred, f'Predicted $\sigma_y(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 3)
    plot_abs_error(x, y, u4_pred, u4_true)
    plt.tight_layout()
    plt.axis('square')
    plt.show()

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plot_sol_2d(x, y, u5_true, r'Exact $\tau_{xy}(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 2)
    plot_sol_2d(x, y, u5_pred, r'Predicted $\tau_{xy}(x,y)$')
    plt.axis('square')

    plt.subplot(1, 3, 3)
    plot_abs_error(x, y, u5_pred, u5_true)
    plt.tight_layout()
    plt.axis('square')
    plt.show()

def plot_abs_error(x, y, up, ue):
    plt.pcolormesh(x, y, torch.abs(up - ue), cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Absolute Error')

def plot_sol_2d(x, y, u, title):
    plt.pcolormesh(x, y, u, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(title)