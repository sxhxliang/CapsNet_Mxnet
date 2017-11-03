import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn


class CapsBlock(nn.Block):
    def __init__(self, dim_vector,n_channels,kernel_size,padding,strides=(1,1), **kwargs):
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
        out = nd.concat(conv1, conv2, conv3, conv4,conv5,conv6,conv7,conv8,conv9,conv10,conv11,conv12,conv13,conv14,conv15,conv16,conv17,conv18,conv19,conv20,conv21,conv22,conv23,conv24,conv25,conv26,conv27,conv28,conv29,conv30,conv31,conv32,dim=3)

       	print(out.shape)
        return out


