"""
MxNet implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.

Usage:
       python CapsNet.py
       
Result:

"""
import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn,Trainer
from CapsBlock import CapsBlock
from CapsLayers import CapsuleLayer, PrimaryCap, Length
import utils



def CapsNet(batch_size, ctx):

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=256, kernel_size=9, strides=1, padding=(0,0), activation='relu'))
        # net.add(CapsBlock(dim_vector=8, n_channels=32, kernel_size=9, strides=2,context=ctx,padding=(0,0)))
        net.add(PrimaryCap(dim_vector=8, n_channels=32, kernel_size=9, batch_size=batch_size,strides=2,context=ctx,padding=(0,0)))
        net.add(CapsuleLayer(num_capsule=10, dim_vector=16, context=ctx, batch_size=batch_size))
        net.add(Length())

    net.initialize(ctx=ctx, init=init.Xavier())
    return net



def margin_loss(y_pred,y_true):
    if len(y_true.shape) != 2 :
        y_true = nd.one_hot(y_true,10)

    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))

    return nd.mean(nd.sum(L, 1))

def totalLoss(y_pred, y_true):
    return margin_loss(y_pred, y_true)+L2Loss(y_pred, y_true)
    # L2Loss = nn.gluon.loss.L2Loss()
    
def L2Loss(y_pred, y_true):
    L2Loss = nn.gluon.loss.L2Loss()
    return L2Loss(y_pred,y_true)

    # return - nd.pick(nd.log(y_pred), y_true)

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train', default=False, type=bool)
    args = parser.parse_args()
    print(args)

    ctx = utils.try_gpu()
    # ctx = mx.cpu()

    train_data, test_data = utils.load_data_mnist(batch_size=args.batch_size,resize=28)
    
    net = CapsNet(batch_size=args.batch_size,ctx=ctx)
    # mx.viz.plot_network(net)
    
    print('====================================net====================================')
    print(net)
    
    if args.train:
        print('====================================train====================================')
        
        trainer = Trainer(net.collect_params(),'adam', {'learning_rate': 0.01})

        utils.train(train_data, test_data, net, margin_loss, trainer, ctx, num_epochs=args.epochs)
