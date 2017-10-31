import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn

from capsulelayers import CapsuleLayer, PrimaryCap, Length
import utils


net = nn.Sequential()
batch_size =2
net.add(PrimaryCap(dim_vector=8, n_channels=32, kernel_size=(9,9), strides=2,padding=(0,0)))
net.add(CapsuleLayer(num_capsule=10, dim_vector=16, batch_size=batch_size))
net.add(Length())
net.initialize()
ctx = mx.cpu()


train_data, test_data = utils.load_data_mnist(batch_size=2,resize=20)

for imgs, _ in train_data:
	break
y = net(imgs)
print('net',net)
print(y.shape)


