import torch
from .losses import LpLoss
from tqdm import tqdm
from .utils import save_checkpoint, save_loss

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
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    train_loss_epoch = torch.zeros(config['train']['epochs'])

    for e in pbar:
        model.train()
        # train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            data_loss = myloss(out, y)
            total_loss = data_loss * data_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_loss += total_loss.item()

        scheduler.step()
        data_l2 /= len(train_loader)
        train_loss /= len(train_loader)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
                    # f'train f error: {train_pino:.5f}; '
                    f'data l2 error: {data_l2:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    # 'Train f error': train_pino,
                    'Train L2 error': data_l2,
                    'Train loss': train_loss,
                }
            )

        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)

        train_loss_epoch[e] = train_loss

    save_loss(config['train']['save_dir'],
              config['train']['loss_save_name'],
              train_loss_epoch)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)