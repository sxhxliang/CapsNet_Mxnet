import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn,Trainer,loss
from CapsLayers import CapsuleLayer, PrimaryCap, Length
import utils

net = nn.Sequential()
batch_size =2
ctx = mx.cpu()
net.add(nn.Conv2D(channels=256, kernel_size=9, strides=1, padding=(0,0), activation='relu'))
net.add(PrimaryCap(dim_vector=8, n_channels=32, kernel_size=9, strides=2,context=ctx,padding=(0,0)))
net.add(CapsuleLayer(num_capsule=10, dim_vector=16,context=ctx, batch_size=batch_size))
net.add(Length())
net.initialize()






# train_data, test_data = utils.load_data_mnist(batch_size=2,resize=20)


# for data, label in train_data:
    # break
data = nd.random_normal(shape=(2,1,28,28))
with autograd.record():
	output = net(data)
	print('output',output)


       