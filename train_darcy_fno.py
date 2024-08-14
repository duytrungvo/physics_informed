"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
from timeit import default_timer
from train_utils.utilities3 import *
from models.utils import _get_act, add_padding2, remove_padding2
# from models import FNO2d1
from train_utils.datasets import DarcyFlow1
from train_utils.utils import save_loss, save_checkpoint

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d1(nn.Module):
    def __init__(self, modes1, modes2,
                 width=32, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 act='gelu',
                 pad_ratio=[0., 0.]):
        super(FNO2d1, self).__init__()
        """
        Args:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to both sides. pad_ratio[0] pads left, pad_ratio[1] pads right. 
        """
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions'
        self.modes1 = modes1
        self.modes2 = modes2

        self.pad_ratio = pad_ratio
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.mlp = nn.ModuleList([MLP(in_size, out_size, mid_size)
                                  for in_size, out_size, mid_size
                                  in zip(self.layers, self.layers[1:], self.layers[1:])])

        self.ws = nn.ModuleList([nn.Conv2d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = MLP(layers[-1], out_dim, fc_dim)
        # referred to: neuraloperator by @author: Zongyi Li (master/fourier_2d.py)
        self.act = _get_act(act)

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        size_1, size_2 = x.shape[1], x.shape[2]
        if max(self.pad_ratio) > 0:
            num_pad1 = [round(i * size_1) for i in self.pad_ratio]
            num_pad2 = [round(i * size_2) for i in self.pad_ratio]
        else:
            num_pad1 = num_pad2 = [0.]

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # B, C, X, Y
        x = add_padding2(x, num_pad1, num_pad2)

        for i, (speconv, mlp, w) in enumerate(zip(self.sp_convs, self.mlp, self.ws)):
            x1 = speconv(x)
            x1 = mlp(x1)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding2(x, num_pad1, num_pad2)
        x = self.fc1(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d2(nn.Module):
    def __init__(self, modes1, modes2,
                 width=32, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 act='gelu',
                 pad_ratio=[0., 0.]):
        super(FNO2d2, self).__init__()
        """
        Args:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to both sides. pad_ratio[0] pads left, pad_ratio[1] pads right. 
        """
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions'
        self.modes1 = modes1
        self.modes2 = modes2

        self.pad_ratio = pad_ratio
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])


        self.ws = nn.ModuleList([nn.Conv2d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        # referred to: neuraloperator by @author: Zongyi Li (master/fourier_2d.py)
        self.act = _get_act(act)

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        size_1, size_2 = x.shape[1], x.shape[2]
        if max(self.pad_ratio) > 0:
            num_pad1 = [round(i * size_1) for i in self.pad_ratio]
            num_pad2 = [round(i * size_2) for i in self.pad_ratio]
        else:
            num_pad1 = num_pad2 = [0.]

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # B, C, X, Y
        x = add_padding2(x, num_pad1, num_pad2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding2(x, num_pad1, num_pad2)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


if __name__ == '__main__':
    ################################################################
    # configs
    ################################################################
    TRAIN_PATH = '/home/vdtrung/trungvd/data/physics_informed/darcy/piececonst_r421_N1024_smooth1.mat'
    TEST_PATH = '/home/vdtrung/trungvd/data/physics_informed/darcy/piececonst_r421_N1024_smooth2.mat'

    normalizer = False
    ntrain = 1000
    ntest = 200

    batch_size = 20
    learning_rate = 0.001
    epochs = 500
    iterations = epochs*(ntrain//batch_size)

    modes = 12
    width = 32

    r = 5
    h = int(((421 - 1)/r) + 1)
    s = h

    ################################################################
    # load data and data normalization
    ################################################################

    train_dataset = DarcyFlow1(TRAIN_PATH, nx=421, sub=r, offset=0, num=ntrain, normalizer=normalizer)
    train_loader = train_dataset.make_loader(batch_size)

    # test_dataset = DarcyFlow1(TEST_PATH, nx=421, sub=r, offset=0, num=ntest)
    # test_loader = test_dataset.make_loader(batch_size)

    reader = MatReader(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

    if normalizer:
        x_test = train_dataset.x_normalizer.encode(x_test)
    x_test = x_test.reshape(ntest, s, s, 1)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    # model = FNO2d(modes, modes, width).to(device)

    # model = FNO2d1(modes1=[12, 12, 12, 12], modes2=[12, 12, 12, 12],
    #                fc_dim=128, layers=[32, 32, 32, 32, 32], pad_ratio=[0.0, 0.11])

    model = FNO2d2(modes1=[12, 12, 12, 12], modes2=[12, 12, 12, 12],
                   fc_dim=128, layers=[32, 32, 32, 32, 32], pad_ratio=[0.0, 0.0])

    print(count_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
    #                  lr=0.001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=[100, 200, 300, 400],
    #                                                  gamma=0.5)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    myloss = LpLoss(size_average=True)
    train_dataset.y_normalizer.to(device)
    train_loss_epoch = torch.zeros(epochs, 1)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s, s)
            if normalizer:
                out = train_dataset.y_normalizer.decode(out)

            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x).reshape(batch_size, s, s)
                if normalizer:
                    out = train_dataset.y_normalizer.decode(out)

                test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

        # train_l2 /= ntrain
        # test_l2 /= ntest
        train_l2 /= (ntrain // batch_size)
        test_l2 /= (ntest // batch_size)
        train_loss_epoch[ep, 0] = train_l2
        t2 = default_timer()
        if (ep % 1) == 0:
            print(ep, t2-t1, train_l2, test_l2)

    save_loss('darcy-FDM',
              'train_loss_history.txt',
              train_loss_epoch)
    save_checkpoint('darcy-FDM',
                    'darcy-pretrain-pino.pt',
                    model, optimizer)