from mxnet import nd
from mxnet.gluon import nn,Parameter
from mxnet import init


def squash(vectors,axis):
    s_squared_norm = nd.sum(nd.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / nd.sqrt(s_squared_norm)
    return scale * vectors

class PrimaryCap(nn.Block):
    def __init__(self,dim_vector,n_channels,kernel_size,padding,context,strides=(1,1),**kwargs):
        super(PrimaryCap, self).__init__(**kwargs)
        # self.squash = squash()
        # self.net = nn.Sequential()

        self.dim_vector = dim_vector
        self.n_channels=n_channels
        self.conv_vector = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        # with self.name_scope():
        #     self.net.add(nn.Conv2D(channels=dim_vector,kernel_size=kernel_size,strides=strides,padding=padding,activation="relu"))
        self.caps = [self.conv_vector for x in range(self.n_channels)]

    def forward(self, x):
        # print('PrimaryCap inputs shape',x.shape)
        # print('len',len(self.caps))

        # outputs = []
        # global out

        for i in range(self.n_channels):
            output = self.caps[i](x)
            # print('output',output)
            # outputs.append(output)
            if i == 0:
                out = nd.concat(nd.reshape(data=output,shape=(-1, self.dim_vector,output.shape[2] ** 2)),dim=2)
            else:

                out = nd.concat(out,nd.reshape(data=output,shape=(-1, self.dim_vector,output.shape[2] ** 2)),dim=2)
   
        # print('PrimaryCap outputs',out.shape)
        
        # outputs = nd.concatenate(outputs, axis=2)
        
        # squash
        v_primary = squash(out,axis=1)
        # print('concatenate outputs',v_primary.shape)
        return v_primary


class CapsuleLayer(nn.Block):
    def __init__(self,num_capsule,dim_vector, batch_size,context,num_routing=3,**kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule #10
        self.dim_vector = dim_vector #16
        self.batch_size = batch_size 
        self.num_routing = num_routing #3

        self.input_num_capsule = 1152 
        self.input_dim_vector = 8

        #  (1152,8,10,16)
        self.W_ij  = self.params.get(
            'weight',shape=(
                self.batch_size,
                self.input_dim_vector,
                self.input_num_capsule,
                self.num_capsule,
                self.dim_vector),init=init.Normal(0.5)) 
                #init.Xavier()
        # self.bias  = self.params.get(
            # 'bias',shape=(batch_size,1,self.input_num_capsule,self.num_capsule,1),init=init.Zero()) 
        
        # self.bias = Parameter('bias', shape=(batch_size,1,self.input_num_capsule,self.num_capsule,1), init=init.Zero())
        self.bias = nd.zeros(shape=(batch_size,1,self.input_num_capsule,self.num_capsule,1),ctx=context)
        # self.bias.initialize(ctx=context)
        # nd.stop_gradient(self.bias.data())
  
    def forward(self, x):
        # print('CapsuleLayer inputs shape',x.shape)
        # self.bias.set_data(nd.stop_gradient(nd.softmax(self.bias.data(), axis=3)))
        # nd.stop_gradient(self.bias.set_data())
        self.bias = nd.zeros(shape=(self.batch_size,1,self.input_num_capsule,self.num_capsule,1),ctx=x.context)
        self.bias = nd.softmax(self.bias, axis=3)


        u = nd.expand_dims(nd.expand_dims(x, 3), 3)
        # u = u.reshape(u,shape=(-1,8,6,6,32))
        u_ = nd.sum(u*self.W_ij.data(),axis=1,keepdims=True)
        s = nd.sum(u_*self.bias,axis=2,keepdims=True)
        v = squash(s,axis=-1)

        for i in range(self.num_routing):

            self.bias = self.bias + nd.sum(u_*v,axis=-1,keepdims=True)

            c =  nd.softmax(self.bias, axis=3)
            s =  nd.sum(u_ * c, axis=2, keepdims=True)
            v = squash(s,axis=-1)
        # print(x.shape)
        # print(u.shape)
        # print(u_.shape)
        # print(s.shape)
        # print(v.shape)
        # print(self.bias.data().shape)
        # print(v.shape)
 
        return nd.reshape(v,shape=(-1,self.num_capsule, self.dim_vector))


class Length(nn.Block):
    def __init__(self, **kwargs):
        super(Length, self).__init__(**kwargs)

    def forward(self, x):
        x = nd.sqrt(nd.sum(nd.square(x), 2))
        # print('Length output shape',x.shape)
        return x

