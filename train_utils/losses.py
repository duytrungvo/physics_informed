import numpy as np
import torch
import torch.nn.functional as F
# from torchmetrics.functional.regression import r2_score
from train_utils.utils import boundary_function, first_order_derivative

def pino_loss_1d(func):
    def inner(*args, **kwargs):
        Du, boundary_l, boundary_r = func(*args, **kwargs)
        f = torch.zeros(Du.shape, device=args[2].device)
        loss_f = F.mse_loss(Du, f)

        loss_boundary_l = F.mse_loss(boundary_l, torch.zeros(boundary_l.shape, device=args[2].device))
        loss_boundary_r = F.mse_loss(boundary_r, torch.zeros(boundary_r.shape, device=args[2].device))

        return loss_f, loss_boundary_l, loss_boundary_r
    return inner

def pino_loss_reduced_order2_1d(func):
    def inner(*args, **kwargs):
        Du1, Du2, boundary_l, boundary_r = func(*args, **kwargs)
        f1 = torch.zeros(Du1.shape, device=args[2].device)
        f_loss1 = F.mse_loss(Du1, f1)
        f2 = torch.zeros(Du2.shape, device=args[2].device)
        f_loss2 = F.mse_loss(Du2, f2)

        loss_boundary_l = F.mse_loss(boundary_l, torch.zeros(boundary_l.shape, device=args[2].device))
        loss_boundary_r = F.mse_loss(boundary_r, torch.zeros(boundary_r.shape, device=args[2].device))

        return f_loss1, f_loss2, loss_boundary_l, loss_boundary_r
    return inner

def pino_loss_FGTimoshenko_beam(func):
    def inner(*args, **kwargs):
        Du1, Du2, Du3, boundary_l, boundary_r = func(*args, **kwargs)
        f1 = torch.zeros(Du1.shape, device=args[2].device)
        f_loss1 = F.mse_loss(Du1, f1)
        f2 = torch.zeros(Du2.shape, device=args[2].device)
        f_loss2 = F.mse_loss(Du2, f2)
        f3 = torch.zeros(Du3.shape, device=args[2].device)
        f_loss3 = F.mse_loss(Du3, f3)

        loss_boundary_l = F.mse_loss(boundary_l, torch.zeros(boundary_l.shape, device=args[2].device))
        loss_boundary_r = F.mse_loss(boundary_r, torch.zeros(boundary_r.shape, device=args[2].device))

        return f_loss1, f_loss2, f_loss3, loss_boundary_l, loss_boundary_r
    return inner

def FDM_Darcy(u, a, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    # ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    # ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    # uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    # uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    # u = u[:, 1:-1, 1:-1]
    # Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = torch.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = torch.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)
    return Du


def darcy_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    # index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
    #                      torch.zeros(size)], dim=0).long()
    # index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
    #                      torch.tensor(range(0, size))], dim=0).long()

    # boundary_u = u[:, index_x, index_y]
    # truth_u = torch.zeros(boundary_u.shape, device=u.device)
    # loss_u = lploss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss.rel(Du, f)

    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = torch.mean(loss_f)
    return loss_f


def FDM_NS_vorticity(w, v=1/40, t_interval=1.0):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])

    dt = t_interval / (nt-1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = wt + (ux*wx + uy*wy - v*wlap)[...,1:-1] #- forcing
    return Du1


def Autograd_Burgers(u, grid, v=1/100):
    from torch.autograd import grad
    gridt, gridx = grid

    ut = grad(u.sum(), gridt, create_graph=True)[0]
    ux = grad(u.sum(), gridx, create_graph=True)[0]
    uxx = grad(ux.sum(), gridx, create_graph=True)[0]
    Du = ut + ux*u - v*uxx
    return Du, ux, uxx, ut


def AD_loss(u, u0, grid, index_ic=None, p=None, q=None):
    batchsize = u.size(0)
    # lploss = LpLoss(size_average=True)

    Du, ux, uxx, ut = Autograd_Burgers(u, grid)

    if index_ic is None:
        # u in on a uniform grid
        nt = u.size(1)
        nx = u.size(2)
        u = u.reshape(batchsize, nt, nx)

        index_t = torch.zeros(nx,).long()
        index_x = torch.tensor(range(nx)).long()
        boundary_u = u[:, index_t, index_x]

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss(ux[:, :, 0], ux[:, :, -1])
    else:
        # u is randomly sampled, 0:p are BC, p:2p are ic, 2p:2p+q are interior
        boundary_u = u[:, :p]
        batch_index = torch.tensor(range(batchsize)).reshape(batchsize, 1).repeat(1, p)
        u0 = u0[batch_index, index_ic]

        # loss_bc0 = F.mse_loss(u[:, p:p+p//2], u[:, p+p//2:2*p])
        # loss_bc1 = F.mse_loss(ux[:, p:p+p//2], ux[:, p+p//2:2*p])

    loss_ic = F.mse_loss(boundary_u, u0)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    return loss_ic, loss_f


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def FDM_Burgers(u, v, D=1):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx)

    u_h = torch.fft.fft(u, dim=2)
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,nx)
    ux_h = 2j *np.pi*k_x*u_h
    uxx_h = 2j *np.pi*k_x*ux_h
    ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux*u - v*uxx)[:,1:-1,:]
    return Du


def PINO_loss(u, u0, v):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    index_t = torch.zeros(nx,).long()
    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    Du = FDM_Burgers(u, v)[:, :, :]
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
    #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_u, loss_f


def PINO_loss3d(u, u0, forcing, v=1/40, t_interval=1.0):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u, v, t_interval)
    f = forcing.repeat(batchsize, 1, 1, nt-2)
    loss_f = lploss(Du, f)

    return loss_ic, loss_f


def PDELoss(model, x, t, nu):
    '''
    Compute the residual of PDE:
        residual = u_t + u * u_x - nu * u_{xx} : (N,1)

    Params:
        - model
        - x, t: (x, t) pairs, (N, 2) tensor
        - nu: constant of PDE
    Return:
        - mean of residual : scalar
    '''
    u = model(torch.cat([x, t], dim=1))
    # First backward to compute u_x (shape: N x 1), u_t (shape: N x 1)
    grad_x, grad_t = torch.autograd.grad(outputs=[u.sum()], inputs=[x, t], create_graph=True)
    # Second backward to compute u_{xx} (shape N x 1)

    gradgrad_x, = torch.autograd.grad(outputs=[grad_x.sum()], inputs=[x], create_graph=True)

    residual = grad_t + u * grad_x - nu * gradgrad_x
    return residual


def get_forcing(S):
    x1 = torch.tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=torch.float).reshape(S, 1).repeat(1, S)
    x2 = torch.tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=torch.float).reshape(1, S).repeat(S, 1)
    return -4 * (torch.cos(4*(x2))).reshape(1,S,S,1)

@pino_loss_1d
def FDM_ElasticBar_Order2(config_data, a, u):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    P0 = config_data['P0']
    E = config_data['E']
    u = u.reshape(batchsize, nx)

    ux = (-1/2 * u[:, :-2] + 1/2 * u[:, 2:]) / dx
    uxx = (u[:, :-2] - 2 * u[:, 1:-1] + u[:, 2:]) / dx ** 2
    ax = (-1/2 * a[:, :-2] + 1/2 * a[:, 2:]) / dx

    Du = ax * ux + a[:, 1:-1] * uxx

    boundary_l = u[:, 0]
    boundary_r = a[:, -1] * (3 / 2 * u[:, -1] - 2 * u[:, -2] + 1 / 2 * u[:, -3]) / dx - P0 / E
    boundary_l = boundary_l.reshape(batchsize, 1)
    boundary_r = boundary_r.reshape(batchsize, 1)

    return Du, boundary_l, boundary_r

@pino_loss_1d
def FDM_ElasticBar_Order4(config_data, a, u):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    P0 = config_data['P0']
    E = config_data['E']
    u = u.reshape(batchsize, nx)

    ux = torch.zeros(batchsize, nx - 2)
    uxx = torch.zeros(batchsize, nx - 2)
    ax = torch.zeros(batchsize, nx - 2)

    ux[:, 1:-1] = (1 / 12 * u[:, :-4] - 2 / 3 * u[:, 1:-3] + 2 / 3 * u[:, 3:-1] - 1 / 12 * u[:, 4:]) / dx
    ux[:, 0] = (-25 / 12 * u[:, 1] + 4 * u[:, 2] - 3 * u[:, 3] + 4 / 3 * u[:, 4] - 1 / 4 * u[:, 5]) / dx
    ux[:, -1] = (25 / 12 * u[:, -2] - 4 * u[:, -3] + 3 * u[:, -4] - 4 / 3 * u[:, -5] + 1 / 4 * u[:, -6]) / dx

    uxx[:, 1:-1] = (-1 / 12 * u[:, :-4]
                    + 4 / 3 * u[:, 1:-3]
                    - 5 / 2 * u[:, 2:-2]
                    + 4 / 3 * u[:, 3:-1]
                    - 1 / 12 * u[:, 4:]) / dx ** 2
    uxx[:, 0] = (15 / 4 * u[:, 1]
                 - 77 / 6 * u[:, 2]
                 + 107 / 6 * u[:, 3]
                 - 13 * u[:, 4]
                 + 61 / 12 * u[:, 5]
                 - 5 / 6 * u[:, 6]) / dx ** 2
    uxx[:, -1] = (15 / 4 * u[:, -2]
                  - 77 / 6 * u[:, -3]
                  + 107 / 6 * u[:, -4]
                  - 13 * u[:, -5]
                  + 61 / 12 * u[:, -6]
                  - 5 / 6 * u[:, -7]) / dx ** 2

    ax[:, 1:-1] = (1 / 12 * a[:, :-4] - 2 / 3 * a[:, 1:-3] + 2 / 3 * a[:, 3:-1] - 1 / 12 * a[:, 4:]) / dx
    ax[:, 0] = (-25 / 12 * a[:, 1] + 4 * a[:, 2] - 3 * a[:, 3] + 4 / 3 * a[:, 4] - 1 / 4 * a[:, 5]) / dx
    ax[:, -1] = (25 / 12 * a[:, -2] - 4 * a[:, -3] + 3 * a[:, -4] - 4 / 3 * a[:, -5] + 1 / 4 * a[:, -6]) / dx

    Du = ax * ux + a[:, 1:-1] * uxx

    boundary_l = u[:, 0]
    boundary_r = a[:, -1] * (25 / 12 * u[:, -1]
                              - 4 * u[:, -2]
                              + 3 * u[:, -3]
                              - 4 / 3 * u[:, -4]
                              + 1 / 4 * u[:, -5]) / dx - P0 / E
    boundary_l = boundary_l.reshape(batchsize, 1)
    boundary_r = boundary_r.reshape(batchsize, 1)

    return Du, boundary_l, boundary_r

@pino_loss_reduced_order2_1d
def FDM_ReducedOrder_ElasticBar(config_data, a, u):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    P0 = config_data['P0']
    E = config_data['E']
    out_dim = config_data['out_dim']
    u = u.reshape(batchsize, nx, out_dim)
    ux = (-1 / 2 * u[:, :-2, 0] + 1 / 2 * u[:, 2:, 0]) / dx
    Du1 = E * a[:, 1:-1, 0] * ux - u[:, 1:-1, 1]
    Du2 = (-1 / 2 * u[:, :-2, 1] + 1 / 2 * u[:, 2:, 1]) / dx

    boundary_l = u[:, 0, 0]
    boundary_r = u[:, -1, 1] - P0
    return Du1, Du2, boundary_l, boundary_r

@pino_loss_reduced_order2_1d
def FDM_ReducedOrder2_Euler_Bernoulli_Beam(config_data, a, u):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    E = config_data['E']
    out_dim = config_data['out_dim']
    u = u.reshape(batchsize, nx, out_dim)

    uxx = (u[:, :-2, 0] - 2 * u[:, 1:-1, 0] + u[:, 2:, 0]) / dx ** 2
    mxx = (u[:, :-2, 1] - 2 * u[:, 1:-1, 1] + u[:, 2:, 1]) / dx ** 2

    Du1 = mxx + a[:, 1:-1, 1]
    Du2 = uxx - u[:, 1:-1, 1] / (E * a[:, 1:-1, 0])

    if config_data['BC'] == 'HH':
        boundary_l = u[:, 0, :]         # w(0) = M(0) = 0
        boundary_r = u[:, -1, :]        # w(L) = M(L) = 0
    if config_data['BC'] == 'CF':
        boundary_l = u[:, 0, 0]         # w(0) = 0
        boundary_r = u[:, -1, 1]        # M(L) = 0

    return Du1, Du2, boundary_l, boundary_r

@pino_loss_reduced_order2_1d
def FDM_ReducedOrder2_Euler_Bernoulli_Beam_BSF(config_data, a, u, bc):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    E = config_data['E']
    BC = config_data['BC']
    out_dim = config_data['out_dim']
    u = u.reshape(batchsize, nx, out_dim)

    mxx = (u[:, :-2, 1] - 2 * u[:, 1:-1, 1] + u[:, 2:, 1]) / dx ** 2
    uxx = (u[:, :-2, 0] - 2 * u[:, 1:-1, 0] + u[:, 2:, 0]) / dx ** 2
    I = a[:, 1:-1, 0]
    q = a[:, 1:-1, 1]
    m = u[:, 1:-1, 1]

    G1, G2, d2G1dx2, d2G2dx2, boundary_l, boundary_r \
        = boundary_function(u[:, :, 0], u[:, :, 1], bc, nx, dx, BC)

    Du1 = mxx - d2G2dx2[:, 1:-1] + q
    # Du2 = E * I * (uxx - d2G1dx2[:, 1:-1]) + m - G2[:, 1:-1]
    Du2 = (uxx - d2G1dx2[:, 1:-1]) - (m - G2[:, 1:-1]) / (E * I)

    return Du1, Du2, boundary_l, boundary_r

@pino_loss_reduced_order2_1d
def zeros_loss(*args, **kwargs):
    return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

@pino_loss_FGTimoshenko_beam
def FDM_FGTimoshenko_Beam_BSF(config_data, a, u):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    q = config_data['q']
    # E = config_data['E']
    BC = config_data['BC']
    out_dim = config_data['out_dim']
    u = u.reshape(batchsize, nx, out_dim)

    u0x = (-u[:, :-2, 0] + u[:, 2:, 0]) / dx / 2
    phix = (-u[:, :-2, 1] + u[:, 2:, 1]) / dx / 2
    wx = (-u[:, :-2, 2] + u[:, 2:, 2]) / dx / 2

    u0xx = (u[:, :-2, 0] - 2 * u[:, 1:-1, 0] + u[:, 2:, 0]) / dx ** 2
    phixx = (u[:, :-2, 1] - 2 * u[:, 1:-1, 1] + u[:, 2:, 1]) / dx ** 2
    wxx = (u[:, :-2, 2] - 2 * u[:, 1:-1, 2] + u[:, 2:, 2]) / dx ** 2

    # E0x = (-a[:, :-2, 0] + a[:, 2:, 0]) / dx / 2
    E1x = (-a[:, :-2, 1] + a[:, 2:, 1]) / dx / 2
    E2x = (-a[:, :-2, 2] + a[:, 2:, 2]) / dx / 2
    G0x = (-a[:, :-2, 3] + a[:, 2:, 3]) / dx / 2

    E0 = a[:, 1:-1, 0]
    E1 = a[:, 1:-1, 1]
    E2 = a[:, 1:-1, 2]
    G0 = a[:, 1:-1, 3]
    # u0 = u[:, 1:-1, 0]
    phi = u[:, 1:-1, 1]
    # w = u[:, 1:-1, 2]

    Du1 = (E2 * phixx + E2x * phix - E1 * u0xx - E1x * u0x - G0 * (wx - phi))
    Du2 = (G0 * wxx + G0x * wx - G0 * phix - G0x * phi + q)
    Du3 = (E0 * u0x - E1 * phix)

    if BC == 'CC':
        boundary_l = u[:, 0, :]
        boundary_r = u[:, -1, :]

    return Du1, Du2, Du3, boundary_l, boundary_r

@pino_loss_reduced_order2_1d
def FDM_FGTimoshenko_Beam_BSF2(config_data, a, u):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    q = config_data['q']
    # E = config_data['E']
    BC = config_data['BC']
    out_dim = config_data['out_dim']
    u = u.reshape(batchsize, nx, out_dim)

    Et = a[:, 1:-1, 0]
    G0 = a[:, 1:-1, 1]
    w = u[:, 1:-1, 0]
    phi = u[:, 1:-1, 1]

    Etx = (-a[:, :-2, 0] + a[:, 2:, 0]) / dx / 2
    G0x = (-a[:, :-2, 1] + a[:, 2:, 1]) / dx / 2

    wx = (-u[:, :-2, 0] + u[:, 2:, 0]) / dx / 2
    phix = (-u[:, :-2, 1] + u[:, 2:, 1]) / dx / 2

    wxx = (u[:, :-2, 0] - 2 * u[:, 1:-1, 0] + u[:, 2:, 0]) / dx ** 2
    phixx = (u[:, :-2, 1] - 2 * u[:, 1:-1, 1] + u[:, 2:, 1]) / dx ** 2

    Du1 = Etx * phix / G0 + Et * phixx / G0 - (wx - phi)
    Du2 = G0x * (wx - phi) / G0 + (wxx - phix) + q / G0

    if BC == 'CC':
        boundary_l = u[:, 0, :]
        boundary_r = u[:, -1, :]

    return Du1, Du2, boundary_l, boundary_r

@pino_loss_reduced_order2_1d
def FDM_FGTimoshenko_Beam_BSF3(config_data, a, u):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    b = config_data['b']
    h = config_data['h']
    kappa = config_data['kappa']
    nu = config_data['nu']
    q = config_data['q']
    # E = config_data['E']
    BC = config_data['BC']
    out_dim = config_data['out_dim']
    u = u.reshape(batchsize, nx, out_dim)

    coef = 6 * kappa / h**2 / (1 + nu)
    Et = a[:, 1:-1, 0]
    G0 = coef * Et
    w = u[:, 1:-1, 0]
    phi = u[:, 1:-1, 1]

    Etx = (-a[:, :-2, 0] + a[:, 2:, 0]) / dx / 2
    G0x = coef * Etx

    wx = (-u[:, :-2, 0] + u[:, 2:, 0]) / dx / 2
    phix = (-u[:, :-2, 1] + u[:, 2:, 1]) / dx / 2

    wxx = (u[:, :-2, 0] - 2 * u[:, 1:-1, 0] + u[:, 2:, 0]) / dx ** 2
    phixx = (u[:, :-2, 1] - 2 * u[:, 1:-1, 1] + u[:, 2:, 1]) / dx ** 2

    # Du1 = (G0x * (wx + phi) + G0 * (wxx + phix)) / (q * b) - 1.0
    # Du2 = (Etx * phix + Et * phixx) - G0 * (wx + phi)

    Du1 = Etx * (wx + phi) + Et * (wxx + phix) - (q * b) / coef
    Du2 = (Etx * phix + Et * phixx) / coef - Et * (wx + phi)

    if BC == 'CC':
        boundary_l = u[:, 0, :]
        boundary_r = u[:, -1, :]

    return Du1, Du2, boundary_l, boundary_r

@pino_loss_reduced_order2_1d
def FDM_ReducedOrder2_Euler_Bernoulli_FGBeam_BSF(config_data, a, u, bc):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    # E = config_data['E']
    q = config_data['q']
    b = config_data['b']
    BC = config_data['BC']
    out_dim = config_data['out_dim']
    u = u.reshape(batchsize, nx, out_dim)

    mxx = (u[:, :-2, 1] - 2 * u[:, 1:-1, 1] + u[:, 2:, 1]) / dx ** 2
    uxx = (u[:, :-2, 0] - 2 * u[:, 1:-1, 0] + u[:, 2:, 0]) / dx ** 2
    D = a[:, 1:-1, 0]
    m = u[:, 1:-1, 1]

    G1, G2, d2G1dx2, d2G2dx2, boundary_l, boundary_r \
        = boundary_function(u[:, :, 0], u[:, :, 1], bc, nx, dx, BC)

    Du1 = (mxx - d2G2dx2[:, 1:-1]) / (q * b) + 1.0
    # Du2 = E * I * (uxx - d2G1dx2[:, 1:-1]) + m - G2[:, 1:-1]
    # Du2 = (uxx - d2G1dx2[:, 1:-1]) - (m - G2[:, 1:-1]) / D
    Du2 = D * (uxx - d2G1dx2[:, 1:-1]) - (m - G2[:, 1:-1])

    return Du1, Du2, boundary_l, boundary_r

@pino_loss_reduced_order2_1d
def FDM_ReducedOrder2_Euler_Bernoulli_FGBeam_BSF_norm(config_data, a, u, bc):
    batchsize = u.size(0)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    L = config_data['L']
    q = config_data['q']
    b = config_data['b']
    h = config_data['h']
    Em = config_data['Em']
    P = 100.0 * Em * h**3 / q / L**4
    BC = config_data['BC']
    out_dim = config_data['out_dim']
    u = u.reshape(batchsize, nx, out_dim)

    mxx = (u[:, :-2, 1] - 2 * u[:, 1:-1, 1] + u[:, 2:, 1]) / dx ** 2
    uxx = (u[:, :-2, 0] - 2 * u[:, 1:-1, 0] + u[:, 2:, 0]) / dx ** 2
    D = a[:, 1:-1, 0]
    m = u[:, 1:-1, 1]

    G1, G2, d2G1dx2, d2G2dx2, boundary_l, boundary_r \
        = boundary_function(u[:, :, 0], u[:, :, 1], bc, nx, dx, BC)

    Du1 = (mxx - d2G2dx2[:, 1:-1]) / (q * b) / L**2 + 1.0
    # Du2 = (uxx - d2G1dx2[:, 1:-1]) - (m - G2[:, 1:-1]) * P * L**2 / D
    Du2 = (uxx - d2G1dx2[:, 1:-1]) * D / P / L**2 - (m - G2[:, 1:-1])

    return Du1, Du2, boundary_l, boundary_r

def FDM_ReducedOrder_2Dplanar(config_data, a, u):
    E = config_data['E']
    nu = config_data['nu']
    D11 = E * (1 - nu) / (1 + nu) / (1 - 2 * nu)
    D12 = E * nu / (1 + nu) / (1 - 2 * nu)
    D21 = D12
    D22 = D11
    D33 = E / 2 / (1 + nu)
    nx = u.size(1)
    dx = 1 / (nx - 1)
    ny = u.size(2)
    dy = 1 / (ny - 1)
    u1 = u[..., 0]      # displ u
    pu1px, pu1py = first_order_derivative(u1, dx, dy)
    u2 = u[..., 1]      # displ v
    pu2px, pu2py = first_order_derivative(u2, dx, dy)
    sx = u[..., 2]      # stres sx
    psxpx, psxpy = first_order_derivative(sx, dx, dy)
    sy = u[..., 3]      # stres sy
    psypx, psypy = first_order_derivative(sy, dx, dy)
    txy = u[..., 4]      # stres txy
    ptxypx, ptxypy = first_order_derivative(txy, dx, dy)
    f = torch.zeros(pu1px.shape, device=u.device)
    fx = a[..., 0]      # body force fx
    fy = a[..., 1]      # body force fy

    pde1 = F.mse_loss(psxpx + ptxypy + fx[:, 1:-1, 1:-1], f)
    pde2 = F.mse_loss(ptxypx + psypy + fy[:, 1:-1, 1:-1], f)
    pde3 = F.mse_loss(sx[:, 1:-1, 1:-1] - D11 * pu1px - D12 * pu2py, f)
    pde4 = F.mse_loss(sy[:, 1:-1, 1:-1] - D21 * pu1px - D22 * pu2py, f)
    pde5 = F.mse_loss(txy[:, 1:-1, 1:-1] - D33 * (pu2px + pu1py), f)
    # lploss = LpLoss(size_average=True)
    # pde1 = lploss.rel(-psxpx - ptxypy, fx[:, 1:-1, 1:-1])
    # pde2 = lploss.rel(-ptxypx - psypy, fy[:, 1:-1, 1:-1])
    # pde3 = lploss.rel(sx[:, 1:-1, 1:-1], D11 * pu1px + D12 * pu2py)
    # pde4 = lploss.rel(sy[:, 1:-1, 1:-1], D21 * pu1px + D22 * pu2py)
    # pde5 = lploss.rel(txy[:, 1:-1, 1:-1], D33 * (pu2px + pu1py))

    # losspde = pde1 + pde2 + pde3 + pde4 + pde5

    return pde1 + pde2 + pde3 + pde4 + pde5
