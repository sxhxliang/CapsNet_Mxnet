from mxnet import nd
from mxnet.gluon import nn,Parameter
from mxnet import init


def squash(vectors,axis):
    s_squared_norm = nd.sum(nd.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / nd.sqrt(s_squared_norm)
    return scale * vectors

class CapsBlock(nn.Block):
    def __init__(self, dim_vector,n_channels,kernel_size,padding,context,strides=(1,1), **kwargs):
        super(CapsBlock, self).__init__(**kwargs)
        self.conv_vector1 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector2 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector3 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector4 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector5 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        
        self.conv_vector6 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector7 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector8 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector9 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector10 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        
        self.conv_vector11 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector12 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector13 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector14 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector15 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        
        self.conv_vector16 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector17 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector18 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector19 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector20 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        
        self.conv_vector21 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector22 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector23 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector24 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector25 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        
        self.conv_vector26 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector27 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector28 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector29 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector30 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
       
        self.conv_vector31 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        self.conv_vector32 = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        
    def forward(self, x):
        conv1 = self.conv_vector1(x)
        # conv1 = nd.reshape(data=conv1,shape=(-1, self.dim_vector,conv1.shape[2] ** 2)))
        # nd.reshape(self.conv_vector2(x),shape=(-1,self.dim_vector,c_shape))
        conv2 = self.conv_vector2(x)
        conv3 = self.conv_vector3(x)
        conv4 = self.conv_vector4(x)
        conv5 = self.conv_vector5(x)

        conv6 = self.conv_vector6(x)
        conv7 = self.conv_vector7(x)
        conv8 = self.conv_vector8(x)
        conv9 = self.conv_vector9(x)
        conv10 = self.conv_vector10(x)

        conv11 = self.conv_vector11(x)
        conv12 = self.conv_vector12(x)
        conv13 = self.conv_vector13(x)
        conv14 = self.conv_vector14(x)
        conv15 = self.conv_vector15(x)

        conv16 = self.conv_vector16(x)
        conv17 = self.conv_vector17(x)
        conv18 = self.conv_vector18(x)
        conv19 = self.conv_vector19(x)
        conv20 = self.conv_vector20(x)


        conv21 = self.conv_vector21(x)
        conv22 = self.conv_vector22(x)
        conv23 = self.conv_vector23(x)
        conv24 = self.conv_vector24(x)
        conv25 = self.conv_vector25(x)

        conv26 = self.conv_vector26(x)
        conv27 = self.conv_vector27(x)
        conv28 = self.conv_vector28(x)
        conv29 = self.conv_vector29(x)
        conv30 = self.conv_vector30(x)


        conv31 = self.conv_vector31(x)
        conv32 = self.conv_vector32(x)
        out = nd.concat(conv1, conv2, conv3, conv4,conv5,conv6,conv7,conv8,conv9,conv10,conv11,conv12,conv13,conv14,conv15,conv16,conv17,conv18,conv19,conv20,conv21,conv22,conv23,conv24,conv25,conv26,conv27,conv28,conv29,conv30,conv31,conv32,dim=1)

        print(out.shape)
        return out




class PrimaryCap(nn.Block):
    def __init__(self,dim_vector,n_channels,kernel_size,padding,context,batch_size,strides=(1,1),**kwargs):
        super(PrimaryCap, self).__init__(**kwargs)
        # self.squash = squash()
        # self.net = nn.Sequential()
        self.batch_size=batch_size
        self.kernel_size = kernel_size
        self.dim_vector = dim_vector
        self.n_channels=n_channels
        self.conv_caps = CapsBlock(dim_vector,n_channels,kernel_size,padding,context,strides)
        # self.conv_vector = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
        # with self.name_scope():
        #     self.net.add(nn.Conv2D(channels=dim_vector,kernel_size=kernel_size,strides=strides,padding=padding,activation="relu"))
        # self.caps = [self.conv_vector for x in range(self.n_channels)]

    def forward(self, x):
        # print('PrimaryCap inputs shape',x.shape)
        # print('len',len(self.caps))

        # outputs = []
        # global out
        x = self.conv_caps(x)
        x = nd.reshape(x,shape=(self.batch_size,self.dim_vector,32,6,6))
        x = nd.reshape(x,shape=(self.batch_size,self.dim_vector,1152))
        # for i in range(self.n_channels):
        #     output = self.caps[i](x)
        #     # print('output',output)
        #     # outputs.append(output)
        #     if i == 0:
        #         out = nd.concat(nd.reshape(data=output,shape=(-1, self.dim_vector,output.shape[2] ** 2)),dim=2)
        #     else:

        #         out = nd.concat(out,nd.reshape(data=output,shape=(-1, self.dim_vector,output.shape[2] ** 2)),dim=2)
   
        # print('PrimaryCap outputs',out.shape)
        
        # outputs = nd.concatenate(outputs, axis=2)
        
        # squash
        v_primary = squash(x,axis=1)
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
        
        self.bias = Parameter('bias', shape=(batch_size,1,self.input_num_capsule,self.num_capsule,1), init=init.Zero(),grad_req='null')
        
        self.bias.initialize(ctx=context)
        self.bias.set_data(nd.softmax(self.bias.data(), axis=3))
        # nd.stop_gradient(self.bias.data())
    def __call_(self,data):
        print(data)
        # pass
  
    def forward(self, x):
        print('CapsuleLayer inputs shape',x.shape)
        print('CapsuleLayer inputs shape',x.shape)
        # self.bias.set_data(nd.stop_gradient(nd.softmax(self.bias.data(), axis=3)))
        # nd.stop_gradient(self.bias.set_data())
        # self.bias = nd.zeros(shape=(self.batch_size,1,self.input_num_capsule,self.num_capsule,1),ctx=x.context)
        # bij = nd.zeros(shape=(batch_size,1,self.input_num_capsule,self.num_capsule,1),ctx=x.context)
        bij = self.bias.data()
        # print(self.W_ij.data().shape)
        # print(self.bias.data().shape)
        u = nd.expand_dims(nd.expand_dims(x, 3), 3)
        # print(u.shape)
        u_ = nd.sum(u*self.W_ij.data(),axis=1,keepdims=True)
        # print(u_.shape)
        s = nd.sum(u_* bij,axis=2,keepdims=True)
        # print(s.shape)
        v = squash(s,axis=-1)

        for i in range(self.num_routing):

            # print('self.bias.data()',self.bias.data())
            bij = bij + nd.sum(u_*v,axis=-1,keepdims=True)
            # self.bias.zero_grad()
            # self.bias.set_data(bias_data)

            c =  nd.stop_gradient(nd.softmax(bij, axis=3))
            s =  nd.sum(u_ * c, axis=2, keepdims=True)
            v = squash(s,axis=-1)
        # print(x.shape)
       
        # self.bias.zero_grad()

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
        return x

 # Decoder network.
class Decoder(nn.Block):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(512, activation='relu'))
            self.net.add(nn.Dense(1024, activation='relu'))
            self.net.add(nn.Dense(784, activation='sigmoid'))


    def forward(self,x):
        digitcaps,label = x
        # assert digitcaps.shape == (-1,16,10)
        # assert digitcaps.shape = (-1,10)
        label = nd.expand_dims(label, 1)
        x = nd.batch_dot(label,digitcaps)
        x = nd.reshape(self.net(x),shape=(1, 28, 28))

        return x
