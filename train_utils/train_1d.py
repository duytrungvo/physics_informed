import torch
from .losses import LpLoss, elastic_bar_loss
from tqdm import tqdm
from .utils import save_checkpoint, save_loss
from softadapt import SoftAdapt, LossWeightedSoftAdapt
from train_utils.aggregator import Relobralo, SoftAdapt, Sum

try:
    import wandb
except ImportError:
    wandb = None

def train_1d(model,
             train_loader,
             optimizer,
             scheduler,
             config,
             log=False,
             use_tqdm=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    bc_weight_l = config['train']['bc_loss_l']
    bc_weight_r = config['train']['bc_loss_r']
    params_loss = {'f_loss', 'bc_loss_l', 'bc_loss_r', 'data_loss'}
    weight_loss = {'f_loss': f_weight, 'bc_loss_l': bc_weight_l, 'bc_loss_r': bc_weight_r, 'data_loss': data_weight}
    E = config['data']['E']
    P0 = config['data']['P0']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    train_loss_epoch = torch.zeros(config['train']['epochs'], 5)

    if config['train']['balance_scheme'] == 'sum':
        aggregator = Sum(params=params_loss, num_losses=len(params_loss),
                           weights=weight_loss)

    if config['train']['balance_scheme'] == 'softadapt':
        aggregator = SoftAdapt(params=params_loss, num_losses=len(params_loss),
                               weights=weight_loss)

    if config['train']['balance_scheme'] == 'relobralo':
       aggregator = Relobralo(params=params_loss, num_losses=len(params_loss),
                              weights=weight_loss)

    count_update_weight = 0

    for e in pbar:
        model.train()
        physic_mse = 0.0
        bc_l_mse = 0.0
        bc_r_mse = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).reshape(y.shape)

            data_loss = myloss(out, y) #torch.tensor([0.0])
            f_loss, bc_loss_l, bc_loss_r = elastic_bar_loss(out, x[:, :, 0], E, P0)

            # balance scheme
            losses = {'f_loss': f_loss, 'bc_loss_l': bc_loss_l, 'bc_loss_r': bc_loss_r, 'data_loss': data_loss}
            total_loss = aggregator(losses, count_update_weight)
            count_update_weight += 1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            physic_mse += f_loss.item()
            bc_l_mse += bc_loss_l.item()
            bc_r_mse += bc_loss_r.item()
            data_l2 += data_loss.item()
            train_loss += total_loss.item()

        scheduler.step()
        physic_mse /= len(train_loader)
        bc_l_mse /= len(train_loader)
        bc_r_mse /= len(train_loader)
        data_l2 /= len(train_loader)
        train_loss /= len(train_loader)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5E} '
                    f'train f error: {physic_mse:.5E}; '
                    f'train bc left error: {bc_l_mse:.5E}; '
                    f'train bc right error: {bc_r_mse:.5E}; '
                    f'data l2 error: {data_l2:.5E}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': physic_mse,
                    'Train bc left error': bc_l_mse,
                    'Train bc right error': bc_r_mse,
                    'Train L2 error': data_l2,
                    'Train loss': train_loss,
                }
            )

        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)

        train_loss_epoch[e, :] = torch.tensor([train_loss, physic_mse, bc_l_mse, bc_r_mse, data_l2])

    save_loss(config['train']['save_dir'],
              config['train']['loss_save_name'],
              train_loss_epoch)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)