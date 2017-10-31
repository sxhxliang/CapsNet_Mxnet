import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn,Trainer,loss

from CapsLayers import CapsuleLayer, PrimaryCap, Length
import utils


net = nn.Sequential()
batch_size =2
net.add(PrimaryCap(dim_vector=8, n_channels=32, kernel_size=9, strides=2,padding=(0,0)))
net.add(CapsuleLayer(num_capsule=10, dim_vector=16, batch_size=batch_size))
net.add(Length())
net.initialize()
ctx = mx.cpu()


def loss(y_pred,y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))


train_data, test_data = utils.load_data_mnist(batch_size=2,resize=20)
trainer = Trainer(net.collect_params(),'adam', {'learning_rate': 0.01})
train_loss = 0
train_acc = 0
# loss = loss.SoftmaxCrossEntropyLoss()
for i, batch in enumerate(train_data):

    data, label = batch
    label = nd.one_hot(label,10)
    label = label.as_in_context(ctx)
    with autograd.record():
        output = net(data)
        print('output',output)


       