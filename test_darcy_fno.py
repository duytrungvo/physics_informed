import torch.nn.functional as F
from timeit import default_timer
from train_utils.utilities3 import *
from models.utils import _get_act
from models import FNO2d2
from train_utils.datasets import DarcyFlow1, DarcyFlow
# from train_darcy_fno import FNO2d2

device = 0 if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
np.random.seed(0)

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
    iterations = epochs * (ntrain // batch_size)

    modes = 12
    width = 32

    r = 5
    h = int(((421 - 1) / r) + 1)
    s = h

    ################################################################
    # load data and data normalization
    ################################################################

    train_dataset = DarcyFlow1(TRAIN_PATH, nx=421, sub=r, offset=0, num=ntrain, normalizer=normalizer)
    # train_loader = train_dataset.make_loader(batch_size)

    # test_dataset = DarcyFlow1(TEST_PATH, nx=421, sub=r, offset=0, num=ntest)
    # test_loader = test_dataset.make_loader(batch_size)

    reader = MatReader(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

    if normalizer:
        x_test = train_dataset.x_normalizer.encode(x_test)
    x_test = x_test.reshape(ntest, s, s, 1)

    test_loader0 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    dataset = DarcyFlow(TEST_PATH,
                        nx=421, sub=r,
                        offset=0, num=ntest)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # model = FNO2d1(modes1=[12, 12, 12, 12], modes2=[12, 12, 12, 12],
    #                fc_dim=128, layers=[32, 32, 32, 32, 32], pad_ratio=[0.0, 0.11])

    # model = FNO2d2(modes1=[12, 12, 12, 12], modes2=[12, 12, 12, 12],
    #                fc_dim=128, layers=[32, 32, 32, 32, 32])

    model = FNO2d2(modes1=[12, 12, 12, 12],
                   modes2=[12, 12, 12, 12],
                   fc_dim=128,
                   layers=[32, 32, 32, 32, 32],
                   act='gelu',
                   pad_ratio=[0.0, 0.0]).to(device)

    myloss = LpLoss(size_average=False)
    if normalizer:
        train_dataset.y_normalizer.to(device)

    # Load from checkpoint
    ckpt_path = 'checkpoints/darcy-FDM/darcy-pretrain-pino.pt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    print('Weights loaded from %s' % ckpt_path)

    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).reshape(batch_size, s, s)
            if normalizer:
                out = train_dataset.y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    test_l2 /= ntest
    print(f'==Averaged relative L2 error mean: {test_l2}==\n')
