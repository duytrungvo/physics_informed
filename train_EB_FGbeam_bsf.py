from argparse import ArgumentParser
import yaml
import torch
from models import FNO1d, FCNet, NNet
from train_utils import Adam
from train_utils.datasets import Loader_1D, Loader_FGbeam
from train_utils.train_1d import train_1drb, train_1d_EB_FGbeam
from train_utils.losses import LpLoss, FDM_ReducedOrder2_Euler_Bernoulli_FGBeam_BSF, \
    FDM_ReducedOrder2_Euler_Bernoulli_FGBeam_BSF_norm
from train_utils.plot_test import plot_pred
from train_utils.utils import shape_function, boundary_function, test_func_disp, test_func_moment
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

torch.manual_seed(0)
np.random.seed(0)

def run(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    dataset = Loader_FGbeam(data_config['datapath'],
                            nx=data_config['nx'],
                            sub=data_config['sub'],
                            in_dim=data_config['in_dim'],
                            out_dim=data_config['out_dim'])

    train_loader = dataset.make_loader(n_sample=data_config['n_sample'],
                                       batch_size=config['train']['batchsize'],
                                       start=data_config['offset'])

    # define model
    model_config = config['model']
    L = data_config['L']
    if model_config['name'] == 'fno':
        model = FNO1d(modes=model_config['modes'],
                      fc_dim=model_config['fc_dim'],
                      layers=model_config['layers'],
                      in_dim=data_config['in_dim'],
                      out_dim=data_config['out_dim'],
                      act=model_config['act']).to(device)

        psi = test_func_disp(data_config['BC'])
        varphi = test_func_moment(data_config['BC'])

        model.apply_output_transform(
            [lambda x, y: psi(x, L) * y,
             lambda x, y: varphi(x, L) * y]
        )

    # train
    train_config = config['train']
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=train_config['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=train_config['milestones'],
                                                     gamma=train_config['scheduler_gamma'])

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)

    pino_loss = FDM_ReducedOrder2_Euler_Bernoulli_FGBeam_BSF
    # pino_loss = FDM_ReducedOrder2_Euler_Bernoulli_FGBeam_BSF_norm
    train_1d_EB_FGbeam(model,
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
    plt.semilogy(range(len(loss_history)), loss_history[:, :5])
    plt.xlabel('Epochs')
    plt.ylabel('$Loss$')
    plt.tight_layout()
    plt.legend(['train_loss', 'pde_mse', 'bc_l_mse', 'bc_r_mse', 'data_l2'])
    plt.show()

def test(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    dataset = Loader_FGbeam(data_config['datapath'],
                            nx=data_config['nx'],
                            sub=data_config['sub'],
                            in_dim=data_config['in_dim'],
                            out_dim=data_config['out_dim'])

    data_loader = dataset.make_loader(n_sample=data_config['n_sample'],
                                      batch_size=config['test']['batchsize'],
                                      start=data_config['offset'],
                                      train=False)

    model_config = config['model']
    L = data_config['L']
    if model_config['name'] == 'fno':
        model = FNO1d(modes=model_config['modes'],
                      fc_dim=model_config['fc_dim'],
                      layers=model_config['layers'],
                      in_dim=data_config['in_dim'],
                      out_dim=data_config['out_dim'],
                      act=model_config['act']).to(device)

        psi = test_func_disp(data_config['BC'])
        varphi = test_func_moment(data_config['BC'])

        model.apply_output_transform(
            [lambda x, y: psi(x, L) * y,
             lambda x, y: varphi(x, L) * y]
        )

    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    pino_loss = FDM_ReducedOrder2_Euler_Bernoulli_FGBeam_BSF
    # pino_loss = FDM_ReducedOrder2_Euler_Bernoulli_FGBeam_BSF_norm
    myloss = LpLoss(size_average=True)
    model.eval()
    nsample = data_config['n_sample']
    in_dim = data_config['in_dim']
    out_dim = data_config['out_dim']
    s = int(np.ceil(data_config['nx'] / data_config['sub']))
    test_x = np.zeros((nsample, s, in_dim))
    preds_y = np.zeros((nsample, s, out_dim))
    test_y = np.zeros((nsample, s, out_dim))
    test_err = []
    test_err_pde1 = []
    test_err_pde2 = []
    batchsize = config['test']['batchsize']
    dx = 1 / (s - 1)
    x_test = data_loader.dataset[0][:, -1].repeat(batchsize).reshape(batchsize, s)
    bc = shape_function(data_config['BC'], x_test, L)
    with torch.no_grad():
        for i, data_x in enumerate(data_loader):
            data_x = data_x.to(device)
            pred_y = model(data_x)

            G1, G2, _, _, _, _ = boundary_function(pred_y[:, :, 0], pred_y[:, :, 1], bc, s, dx, data_config['BC'])

            pred_y0 = pred_y[:, :, 0] - G1
            pred_y1 = pred_y[:, :, 1] - G2
            pred_y_bsf = torch.stack((pred_y0, pred_y1), 2)

            # data_loss = torch.tensor(0)
            pde_loss1, pde_loss2, _, _ = pino_loss(config['data'], data_x, pred_y_bsf, bc)
            pde_loss = pde_loss1 + pde_loss2
            # data_loss_w = myloss(pred_y_bsf[:, :, 0], data_y[:, :, 0])
            # data_loss_m = myloss(pred_y_bsf[:, :, 1], data_y[:, :, 1])
            test_err.append(pde_loss.item())
            test_err_pde1.append(pde_loss1.item())
            test_err_pde2.append(pde_loss2.item())
            test_x[i] = data_x.cpu().numpy()
            # test_y[i] = data_y.cpu().numpy()
            preds_y[i] = pred_y_bsf.cpu().numpy()

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))
    mean_err_pde1 = np.mean(test_err_pde1)
    std_err_pde1 = np.std(test_err_pde1, ddof=1) / np.sqrt(len(test_err))
    mean_err_pde2 = np.mean(test_err_pde2)
    std_err_pde2 = np.std(test_err_pde2, ddof=1) / np.sqrt(len(test_err))

    print(f'==Test for {nsample} samples==')
    print(f'==Averaged relative L2 error mean(pde1 & 2): {mean_err}, std error: {std_err}==')
    print(f'==Averaged relative L2 error mean(pde1): {mean_err_pde1}, std error: {std_err_pde1}==')
    print(f'==Averaged relative L2 error mean(pde2): {mean_err_pde2}, std error: {std_err_pde2}==')

    # savemat('EB_powerlaw_CC_data3.mat', {'x': test_x, 'yp': preds_y})
    # err_idx = np.argsort(test_err)
    # err_idx = np.arange(0, nsample)
    # np.random.shuffle(err_idx)
    # plot_pred(data_config, test_x, test_y, preds_y, err_idx)

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