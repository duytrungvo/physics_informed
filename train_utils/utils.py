import os
import numpy as np
import torch


def vor2vel(w, L=2 * np.pi):
    '''
    Convert vorticity into velocity
    Args:
        w: vorticity with shape (batchsize, num_x, num_y, num_t)

    Returns:
        ux, uy with the same shape
    '''
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(N, 1).repeat(1, N).reshape(1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(1, N).repeat(N, 1).reshape(1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 2 * np.pi / L * 1j * k_y * f_h
    uy_h = -2 * np.pi / L * 1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    return ux, uy


def get_sample(N, T, s, p, q):
    # sample p nodes from Initial Condition, p nodes from Boundary Condition, q nodes from Interior

    # sample IC
    index_ic = torch.randint(s, size=(N, p))
    sample_ic_t = torch.zeros(N, p)
    sample_ic_x = index_ic/s

    # sample BC
    sample_bc = torch.rand(size=(N, p//2))
    sample_bc_t =  torch.cat([sample_bc, sample_bc],dim=1)
    sample_bc_x = torch.cat([torch.zeros(N, p//2), torch.ones(N, p//2)],dim=1)

    # sample I
    # sample_i_t = torch.rand(size=(N,q))
    # sample_i_t = torch.rand(size=(N,q))**2
    sample_i_t = -torch.cos(torch.rand(size=(N, q))*np.pi/2) + 1
    sample_i_x = torch.rand(size=(N,q))

    sample_t = torch.cat([sample_ic_t, sample_bc_t, sample_i_t], dim=1).cuda()
    sample_t.requires_grad = True
    sample_x = torch.cat([sample_ic_x, sample_bc_x, sample_i_x], dim=1).cuda()
    sample_x.requires_grad = True
    sample = torch.stack([sample_t, sample_x], dim=-1).reshape(N, (p+p+q), 2)
    return sample, sample_t, sample_x, index_ic.long()


def get_grid(N, T, s):
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float).reshape(1, T, 1).repeat(N, 1, s).cuda()
    gridt.requires_grad = True
    gridx = torch.tensor(np.linspace(0, 1, s+1)[:-1], dtype=torch.float).reshape(1, 1, s).repeat(N, T, 1).cuda()
    gridx.requires_grad = True
    grid = torch.stack([gridt, gridx], dim=-1).reshape(N, T*s, 2)
    return grid, gridt, gridx


def get_2dgrid(S):
    '''
    get array of points on 2d grid in (0,1)^2
    Args:
        S: resolution

    Returns:
        points: flattened grid, ndarray (N, 2)
    '''
    xarr = np.linspace(0, 1, S)
    yarr = np.linspace(0, 1, S)
    xx, yy = np.meshgrid(xarr, yarr, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel()], axis=0).T
    return points


def torch2dgrid(num_x, num_y, bot=(0,0), top=(1,1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh


def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt


def convert_ic(u0, N, S, T, time_scale=1.0):
    u0 = u0.reshape(N, S, S, 1, 1).repeat([1, 1, 1, T, 1])
    gridx, gridy, gridt = get_grid3d(S, T, time_scale=time_scale, device=u0.device)
    a_data = torch.cat((gridx.repeat([N, 1, 1, 1, 1]), gridy.repeat([N, 1, 1, 1, 1]),
                        gridt.repeat([N, 1, 1, 1, 1]), u0), dim=-1)
    return a_data



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def set_grad(tensors, flag=True):
    for p in tensors:
        p.requires_grad = flag


def zero_grad(params):
    '''
    set grad field to 0
    '''
    if isinstance(params, torch.Tensor):
        if params.grad is not None:
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()


def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count


def save_checkpoint(path, name, model, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    torch.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)



def save_ckpt(path, model, optimizer=None, scheduler=None):
    model_state = model.state_dict()
    if optimizer:
        optim_state = optimizer.state_dict()
    else:
        optim_state = None
    
    if scheduler:
        scheduler_state = scheduler.state_dict()
    else:
        scheduler_state = None
    torch.save({
        'model': model_state, 
        'optim': optim_state, 
        'scheduler': scheduler_state
    }, path)
    print(f'Checkpoint is saved to {path}')


def dict2str(log_dict):
    res = ''
    for key, value in log_dict.items():
        res += f'{key}: {value}|'
    return res

def save_loss(path, name, loss):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    np.savetxt(ckpt_dir + name, loss)
    print('Training lost is saved at %s' % ckpt_dir + name)

def test_func_disp(BC):
    if BC == 'CC':
        return lambda x, l: x * (l - x)                                                 #psi1
        # return lambda x, l: - x ** 2 / l + x ** 3 / l ** 2                            #psi2
        # return lambda x, l: x ** 2 / l ** 2 - 2 * x ** 3 / l ** 3 + x ** 4 / l ** 4   #psi3
        # return lambda x, l: torch.cos(2 * torch.pi * x / l) - 1                       #psi4
    if BC == 'CH':
        # return lambda x, l: x * (l - x)
        return lambda x, l: - x ** 2 / l + x ** 3 / l ** 2
    if BC == 'CF':
        return lambda x, l: x * (l - x)
    if BC == 'HH':
        return lambda x, l: x * (l - x)
def test_func_moment(BC):
    if BC == 'CC':
        return lambda x, l: 1
    if BC == 'CH':
        return lambda x, l: 1 - x / l
        # return lambda x, l: l - x
    if BC == 'CF':
        return lambda x, l: x * (l - x)
    if BC == 'HH':
        return lambda x, l: x * (l - x)

def shape_function(BC, x, L):
    bc = torch.zeros((x.size(0), x.size(1), 16))
    if BC == 'CF':
        bc[:, :, 0] = 1
        bc[:, :, 1] = x
        bc[:, :, 6] = 1
        bc[:, :, 7] = x - L
    if BC == 'HH':
        bc[:, :, 0] = 1 - x / L
        bc[:, :, 2] = x / L
        bc[:, :, 4] = 1 - x / L
        bc[:, :, 6] = x / L
    if BC == 'CH':
        bc[:, :, 0] = 1 - x ** 2 / L ** 2
        bc[:, :, 1] = x - x ** 2 / L
        bc[:, :, 2] = x ** 2 / L ** 2
        bc[:, :, 6] = 1
        bc[:, :, 8] = - 2 / L ** 2
        bc[:, :, 9] = - 2 / L
        bc[:, :, 10] = 2 / L ** 2
    if BC == 'CC':
        bc[:, :, 0] = 1 - 3 * x ** 2 / L ** 2 + 2 * x ** 3 / L ** 3
        bc[:, :, 1] = x - 2 * x ** 2 / L + x ** 3 / L ** 2
        bc[:, :, 2] = 3 * x ** 2 / L ** 2 - 2 * x ** 3 / L ** 3
        bc[:, :, 3] = - x ** 2 / L + x ** 3 / L ** 2
        bc[:, :, 8] = - 6 / L ** 2 + 12 * x / L ** 3
        bc[:, :, 9] = - 4 / L + 6 * x / L ** 2
        bc[:, :, 10] = 6 / L ** 2 - 12 * x / L ** 3
        bc[:, :, 11] = - 2 / L + 6 * x / L ** 2

    return bc
def boundary_function(w, m, bc, nx, dx, BC):
    batchsize = w.size(0)

    b1, c1, d2b1dx2, d2c1dx2 = bc[:, :, 0], bc[:, :, 4], bc[:, :, 8], bc[:, :, 12]
    b2, c2, d2b2dx2, d2c2dx2 = bc[:, :, 1], bc[:, :, 5], bc[:, :, 9], bc[:, :, 13]
    b3, c3, d2b3dx2, d2c3dx2 = bc[:, :, 2], bc[:, :, 6], bc[:, :, 10], bc[:, :, 14]
    b4, c4, d2b4dx2, d2c4dx2 = bc[:, :, 3], bc[:, :, 7], bc[:, :, 11], bc[:, :, 15]

    w0 = torch.repeat_interleave(w[:, 0], nx, dim=0).reshape((batchsize, nx))
    wL = torch.repeat_interleave(w[:, -1], nx, dim=0).reshape((batchsize, nx))

    m0 = torch.repeat_interleave(m[:, 0], nx, dim=0).reshape((batchsize, nx))
    mL = torch.repeat_interleave(m[:, -1], nx, dim=0).reshape((batchsize, nx))

    dw0 = (-1.5 * w[:, 0] + 2 * w[:, 1] - 0.5 * w[:, 2]) / dx
    dwdx0 = torch.repeat_interleave(dw0, nx, dim=0).reshape((batchsize, nx))

    dwL = (0.5 * w[:, -3] - 2 * w[:, -2] + 1.5 * w[:, -1]) / dx
    dwdxL = torch.repeat_interleave(dwL, nx, dim=0).reshape((batchsize, nx))

    dm0 = (-1.5 * m[:, 0] + 2 * m[:, 1] - 0.5 * m[:, 2]) / dx
    dmdx0 = torch.repeat_interleave(dm0, nx, dim=0).reshape((batchsize, nx))

    dmL = (0.5 * m[:, -3] - 2 * m[:, -2] + 1.5 * m[:, -1]) / dx
    dmdxL = torch.repeat_interleave(dmL, nx, dim=0).reshape((batchsize, nx))

    G1 = b1 * w0 + b2 * dwdx0 + b3 * wL + b4 * dwdxL
    G2 = c1 * m0 + c2 * dmdx0 + c3 * mL + c4 * dmdxL
    d2G1dx2 = d2b1dx2 * w0 + d2b2dx2 * dwdx0 + d2b3dx2 * wL + d2b4dx2 * dwdxL
    d2G2dx2 = d2c1dx2 * m0 + d2c2dx2 * dmdx0 + d2c3dx2 * mL + d2c4dx2 * dmdxL

    if BC == 'CF':
        boundary_l = torch.stack((w0, dwdx0), 1)
        boundary_r = torch.stack((mL, dmdxL), 1)
    if BC == 'CH':
        boundary_l = torch.stack((w0, dwdx0), 1)
        boundary_r = torch.stack((wL, mL), 1)
    if BC == 'CC':
        boundary_l = torch.stack((w0, dwdx0), 1)
        boundary_r = torch.stack((wL, dwdxL), 1)
    if BC == 'HH':
        boundary_l = torch.stack((w0, m0), 1)
        boundary_r = torch.stack((wL, mL), 1)

    return G1, G2, d2G1dx2, d2G2dx2, boundary_l, boundary_r