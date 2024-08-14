import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint, save_loss
from .losses import LpLoss, darcy_loss, PINO_loss
# from .utilities3 import LpLoss
from timeit import default_timer

try:
    import wandb
except ImportError:
    wandb = None


def train_2d_operator(model,
                      train_loader,
                      optimizer, scheduler,
                      config,
                      rank=0, log=False,
                      project='PINO-2d-default',
                      group='default',
                      tags=['default'],
                      use_tqdm=True,
                      profile=False):
    '''
    train PINO on Darcy Flow
    Args:
        model:
        train_loader:
        optimizer:
        scheduler:
        config:
        rank:
        log:
        project:
        group:
        tags:
        use_tqdm:
        profile:

    Returns:

    '''
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    mesh = train_loader.dataset.mesh
    mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    mollifier = mollifier.to(rank)
    pde_mesh = train_loader.dataset.pde_mesh
    pde_mol = torch.sin(np.pi * pde_mesh[..., 0]) * torch.sin(np.pi * pde_mesh[..., 1]) * 0.001
    pde_mol = pde_mol.to(rank)
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for data_ic, u, pde_ic in train_loader:
            data_ic, u, pde_ic = data_ic.to(rank), u.to(rank), pde_ic.to(rank)

            optimizer.zero_grad()

            # data loss
            if data_weight > 0:
                pred = model(data_ic).squeeze(dim=-1)
                pred = pred * mollifier
                data_loss = myloss(pred, y)

            a = x[..., 0]
            f_loss = darcy_loss(pred, a)

            loss = data_weight * data_loss + f_weight * f_loss
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item() * y.shape[0]
            loss_dict['f_loss'] += f_loss.item() * y.shape[0]
            loss_dict['data_loss'] += data_loss.item() * y.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    f'data loss: {data_loss_val:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'train loss': train_loss_val,
                    'f loss': f_loss_val,
                    'data loss': data_loss_val
                }
            )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()
    print('Done!')

def train_2d_darcy(model,
                      train_loader,
                      optimizer, scheduler,
                      config,
                      use_tqdm=True):
    '''
    train PINO on Darcy Flow
    Args:
        model:
        train_loader:
        optimizer:
        scheduler:
        config:
        rank:
        log:
        project:
        group:
        tags:
        use_tqdm:
        profile:

    Returns:

    '''

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    nsample = len(train_loader)
    batch_size = train_loader.batch_size
    ntrain = nsample * batch_size
    s = train_loader.dataset.s
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    train_loss_epoch = torch.zeros(config['train']['epochs'], 3)
    for e in pbar:
        model.train()
        t1 = default_timer()
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for x, y in train_loader:

            optimizer.zero_grad()

            # data loss
            pred = model(x).reshape(batch_size, s, s)
            data_loss = myloss(pred.view(batch_size,-1), y.view(batch_size,-1))

            a = x[..., 0]
            f_loss = darcy_loss(pred, a)

            loss = data_weight * data_loss + f_weight * f_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_dict['train_loss'] += loss.item()
            loss_dict['f_loss'] += f_loss.item()
            loss_dict['data_loss'] += data_loss.item()

        model.eval()
        train_loss_val = loss_dict['train_loss'] / nsample
        f_loss_val = loss_dict['f_loss'] / nsample
        data_loss_val = loss_dict['data_loss'] / nsample

        t2 = default_timer()
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.10f}, '
                    f'f_loss: {f_loss_val:.10f}, '
                    f'data loss: {data_loss_val:.10f}'
                )
            )
        else:
            print(e, t2-t1, train_loss_val)

        train_loss_epoch[e, :] = torch.tensor([train_loss_val, f_loss_val, data_loss_val])

    save_loss(config['train']['save_dir'],
              config['train']['loss_save_name'],
              train_loss_epoch)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')

def train_2d_burger(model,
                    train_loader, v,
                    optimizer, scheduler,
                    config,
                    rank=0, log=False,
                    project='PINO-2d-default',
                    group='default',
                    tags=['default'],
                    use_tqdm=True):
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    for e in pbar:
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0

        for x, y in train_loader:
            # x, y = x.to(rank), y.to(rank)
            x, y = x.to(device), y.to(device)
            out = model(x).reshape(y.shape)
            data_loss = myloss(out, y)

            loss_u, loss_f = PINO_loss(out, x[:, 0, :, 0], v)
            total_loss = loss_u * ic_weight + loss_f * f_weight + data_loss * data_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()
        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        train_loss /= len(train_loader)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
                    f'train f error: {train_pino:.5f}; '
                    f'data l2 error: {data_l2:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': train_pino,
                    'Train L2 error': data_l2,
                    'Train loss': train_loss,
                }
            )

        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')


def train_2d_planar(model,
                      train_loader,
                      optimizer, scheduler,
                      config,
                      use_tqdm=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    batch_size = train_loader.batch_size
    nsample = len(train_loader)

    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    train_loss_epoch = torch.zeros(config['train']['epochs'], 3)
    for e in pbar:
        model.train()
        t1 = default_timer()
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}

        for a, u, param in train_loader:

            optimizer.zero_grad()

            # data loss
            pred = model(a).reshape(u.shape)
            data_loss = myloss(pred.view(batch_size, -1), u.view(batch_size, -1))

            f_loss = torch.tensor(0.0, dtype=torch.float)

            loss = data_weight * data_loss + f_weight * f_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_dict['train_loss'] += loss.item()
            loss_dict['f_loss'] += f_loss.item()
            loss_dict['data_loss'] += data_loss.item()


        train_loss_val = loss_dict['train_loss'] / nsample
        f_loss_val = loss_dict['f_loss'] / nsample
        data_loss_val = loss_dict['data_loss'] / nsample

        t2 = default_timer()
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    f'data loss: {data_loss_val:.5f}'
                )
            )
        else:
            print(e, t2-t1, train_loss_val)

        train_loss_epoch[e, :] = torch.tensor([train_loss_val, f_loss_val, data_loss_val])

    save_loss(config['train']['save_dir'],
              config['train']['loss_save_name'],
              train_loss_epoch)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)

    print('Done!')


def train_2d_planar_r(model,
                    train_loader,
                    optimizer, scheduler,
                    config,
                    pino_loss,
                    use_tqdm=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_data = config['data']
    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    # E = config_data['E']
    # nu = config_data['nu']
    # lmbd = E * nu / (1 + nu) / (1 - 2 * nu)
    # mu = E / 2 / (1 + nu)
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    mesh = train_loader.dataset.mesh
    x = mesh[..., 0]
    y = mesh[..., 1]
    train_loss_epoch = torch.zeros(config['train']['epochs'], 3)
    for e in pbar:
        model.train()
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for a, u, Q in train_loader:
            # data_ic, u, pde_ic = data_ic.to(rank), u.to(rank), pde_ic.to(rank)

            # data loss
            pred = model(a).reshape(u.shape)
            pred[..., 0] = pred[..., 0] * y * (1 - y)       # displ u
            pred[..., 1] = pred[..., 1] * x * (1 - x) * y   # displ v
            pred[..., 2] = pred[..., 2] * x * (1 - x) * y * (1 - y)         # stres sx
            pred[:, -1, :, 2] = Q * torch.sin(np.pi * x[-1, :])             # stres sx
            pred[..., 3] = pred[..., 3] * x * (1 - x) * y * (1 - y)         # stres sy
            pred[:, -1, :, 3] = 2 * Q * torch.sin(np.pi * x[-1, :])         # stres sy

            data_loss = myloss(pred, u)

            f_loss = pino_loss(config_data, a, pred)

            loss = data_weight * data_loss + f_weight * f_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item() * u.shape[0]
            loss_dict['f_loss'] += f_loss.item() * u.shape[0]
            loss_dict['data_loss'] += data_loss.item() * u.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    f'data loss: {data_loss_val:.5f}'
                )
            )

        train_loss_epoch[e, :] = torch.tensor([train_loss_val, f_loss_val, data_loss_val])

    save_loss(config['train']['save_dir'],
              config['train']['loss_save_name'],
              train_loss_epoch)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)

    print('Done!')
