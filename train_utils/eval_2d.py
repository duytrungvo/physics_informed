from tqdm import tqdm
import numpy as np

import torch

from .losses import LpLoss, darcy_loss, PINO_loss

from .plot_test import plot_darcy, plot_2Dplanar

try:
    import wandb
except ImportError:
    wandb = None


def eval_darcy(model,
               dataloader,
               config,
               device,
               use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    mesh = dataloader.dataset.mesh
    mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    mollifier = mollifier.to(device)
    f_val = []
    test_err = []

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            pred = model(x).reshape(y.shape)
            pred = pred * mollifier

            data_loss = myloss(pred, y)
            a = x[..., 0]
            f_loss = darcy_loss(pred, a)

            test_err.append(data_loss.item())
            f_val.append(f_loss.item())
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Equation error: {f_loss.item():.5f}, test l2 error: {data_loss.item()}'
                    )
                )
    mean_f_err = np.mean(f_val)
    std_f_err = np.std(f_val, ddof=1) / np.sqrt(len(f_val))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')

    plot_darcy(mesh, x, y, pred)

def eval_darcy1(model,
               dataset,
               config,
               device,
               use_tqdm=True):
    myloss = LpLoss(size_average=True)
    dataloader = dataset.make_loader(batch_size=config['test']['batchsize'])
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    f_val = []
    test_err = []

    model.eval()
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            pred = model(x).reshape(y.shape)
            pred = dataset.y_normalizer.decode(pred)

            data_loss = myloss(pred, y)
            a = x[..., 0]
            f_loss = darcy_loss(pred, a)

            test_err.append(data_loss.item())
            f_val.append(f_loss.item())
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Equation error: {f_loss.item():.5f}, test l2 error: {data_loss.item()}'
                    )
                )
    mean_f_err = np.mean(f_val)
    std_f_err = np.std(f_val, ddof=1) / np.sqrt(len(f_val))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')

    # plot_darcy(mesh, x, y, pred)

def eval_burgers(model,
                 dataloader,
                 v,
                 config,
                 device,
                 use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    f_err = []

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x).reshape(y.shape)
        data_loss = myloss(out, y)

        loss_u, f_loss = PINO_loss(out, x[:, 0, :, 0], v)
        test_err.append(data_loss.item())
        f_err.append(f_loss.item())

    mean_f_err = np.mean(f_err)
    std_f_err = np.std(f_err, ddof=1) / np.sqrt(len(f_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')


def eval_2Dplanar(model,
               dataloader,
               config,
               device,
               use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    mesh = dataloader.dataset.mesh
    # mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    # mollifier = mollifier.to(device)
    f_val = []
    test_err = []

    with torch.no_grad():
        for x, y, p in pbar:
            x, y = x.to(device), y.to(device)

            pred = model(x).reshape(y.shape)
            # pred = pred * mollifier

            data_loss = myloss(pred, y)
            # a = x[..., 0]
            f_loss = torch.tensor(0.0, dtype=torch.float)

            test_err.append(data_loss.item())
            f_val.append(f_loss.item())
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Equation error: {f_loss.item():.5f}, test l2 error: {data_loss.item()}'
                    )
                )
    mean_f_err = np.mean(f_val)
    std_f_err = np.std(f_val, ddof=1) / np.sqrt(len(f_val))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')

    # plot_2Dplanar(mesh, x, y, pred)


def eval_2Dplanar_r(model,
               dataloader,
               config,
               pino_loss,
               device,
               use_tqdm=True):
    E = config['data']['E']
    nu = config['data']['nu']
    lmbd = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    mesh = dataloader.dataset.mesh
    x = mesh[..., 0]
    y = mesh[..., 1]
    # mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    # mollifier = mollifier.to(device)
    f_val = []
    test_err = []

    with torch.no_grad():
        for a, u, Q in pbar:
            a, u, Q = a.to(device), u.to(device), Q.to(device)

            pred = model(a).reshape(u.shape)
            pred[..., 0] = pred[..., 0] * y * (1 - y)           # displ u
            pred[..., 1] = pred[..., 1] * x * (1 - x) * y       # displ v
            pred[..., 2] = pred[..., 2] * x * (1 - x) * y * (1 - y)         # stres sx
            pred[:, -1, :, 2] = Q * torch.sin(np.pi * x[-1, :])             # stres sx
            pred[..., 3] = pred[..., 3] * x * (1 - x) * y * (1 - y)         # stres sy
            pred[:, -1, :, 3] = 2 * Q * torch.sin(np.pi * x[-1, :])         # stres sy

            data_loss = myloss(pred, u)
            f_loss = pino_loss(config['data'], a, pred)

            test_err.append(data_loss.item())
            f_val.append(f_loss.item())
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Equation error: {f_loss.item():.5f}, test l2 error: {data_loss.item()}'
                    )
                )
    mean_f_err = np.mean(f_val)
    std_f_err = np.std(f_val, ddof=1) / np.sqrt(len(f_val))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')

    plot_2Dplanar(mesh, a, u, pred)
