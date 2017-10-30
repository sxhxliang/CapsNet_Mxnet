from mxnet import nd
from mxnet.gluon import nn


def squash(vectors):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = nd.sum(nd.square(vectors), -1, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / nd.sqrt(s_squared_norm)
    return scale * vectors


class Length(nn.Block):
    def __init__(self, **kwargs):
        super(Length, self).__init__(**kwargs)

    def forward(self, x):
        return nd.sqrt(nd.sum(nd.square(x), -1))

class CapsuleLayer(nn.Block):
    def __init__(self,num_capsule,dim_vector, batch_size, num_routing=3,**kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.batch_size = batch_size
        self.num_routing = num_routing

        self.squash = squash()

        self.input_num_capsule = 1152
        self.input_dim_vector = 8

        self.W  = self.params.get(
                'weight',shape=(self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector))
        self.bias = self.params.get('bias', shape=(self.input_num_capsule, self.num_capsule))

    def forward(self, x):
        # inputs.shape=[batch_size, input_num_capsule, input_dim_vector]
        # Expand dims to [batch_size, input_num_capsule, 1, 1, input_dim_vector]
        # print('inputs',inputs.get_shape()) 
        # (?, 1152, 8)
        inputs_expand = nd.expand_dims(nd.expand_dims(inputs, 2), 2)
        # print('inputs_expand',inputs_expand.get_shape())
        # inputs_expand (?, 1152, 1, 1, 8)
        # 
        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [batch_size, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = nd.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
        # print('inputs_tiled',inputs_tiled.get_shape())
        # inputs_tiled (?, 1152, 10, 1, 8)
        # 
        # Prepare the dimension of W
        # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
        w_tiled = nd.tile(nd.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])
        # print('w_tiled',w_tiled.get_shape())
        # w_tiled (100, 1152, 10, 8, 16)
        # Transformed vectors, shape = [batch_size, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = nd.batch_dot(inputs_tiled, w_tiled, [4, 3])
        # print('inputs_hat',inputs_hat.get_shape())
        # inputs_hat (100, 1152, 10, 1, 16)

        # update self.bias by routing algorithm
        for _ in range(self.num_routing):
            c = nd.softmax(self.bias)
            c_expand = nd.expand_dims(nd.expand_dims(nd.expand_dims(c, 2), 2), 0)
            outputs = nd.sum(c_expand * inputs_hat, [1, 3], keepdims=True)
            outputs = squash(outputs)
            self.bias = self.bias + nd.sum(inputs_hat * outputs, [0, -2, -1])
            # nd.update(self.bias, )
        return nd.reshape(outputs, [self.batch_size, self.num_capsule, self.dim_vector])


# class CapsuleLayert(nn.Block):
#         """
#     The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
#     neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
#     from scalar to vector. So its input shape = [batch_size, input_num_capsule, input_dim_vector] and output shape = \
#     [batch_size, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.
    
#     :param num_capsule: number of capsules in this layer
#     :param dim_vector: dimension the output vectors of the capsules in this layer
#     :param batch_size: the batch_size during training. This is wired to require batch_size when defining a Layer. But I
#         have not figured out a better way.
#     :param num_routings: number of iterations for the routing algorithm
#     """
#     def __init__(self,num_capsule,dim_vector, batch_size, num_routing=3,**kwargs):
#         super(CapsuleLayer, self).__init__(**kwargs)

#     # def __init__(self, num_capsule, dim_vector, batch_size, num_routing=3, **kwargs):
#         # super(CapsuleLayer, self).__init__(**kwargs)
#         self.num_capsule = num_capsule
#         self.dim_vector = dim_vector
#         self.batch_size = batch_size
#         self.num_routing = num_routing
#         # self.kernel_initializer = initializers.get(kernel_initializer)
#         # self.bias_initializer = initializers.get(bias_initializer)
#         self.squash = squash()
#         # assert len(input_shape) >= 3
#         # (None, 1152, 8)
#         self.input_num_capsule = 1152
#         self.input_dim_vector = 8

#         self.W  = self.params.get(
#                 'weight',shape=(self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector))
#         self.bias = self.params.get('bias', shape=(self.input_num_capsule, self.num_capsule))

#     def forward(self, x):
#         # inputs.shape=[batch_size, input_num_capsule, input_dim_vector]
#         # Expand dims to [batch_size, input_num_capsule, 1, 1, input_dim_vector]
#         # print('inputs',inputs.get_shape()) 
#         # (?, 1152, 8)
#         inputs_expand = nd.expand_dims(nd.expand_dims(inputs, 2), 2)
#         # print('inputs_expand',inputs_expand.get_shape())
#         # inputs_expand (?, 1152, 1, 1, 8)
#         # 
#         # Replicate num_capsule dimension to prepare being multiplied by W
#         # Now it has shape = [batch_size, input_num_capsule, num_capsule, 1, input_dim_vector]
#         inputs_tiled = nd.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
#         # print('inputs_tiled',inputs_tiled.get_shape())
#         # inputs_tiled (?, 1152, 10, 1, 8)
#         # 
#         # Prepare the dimension of W
#         # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
#         w_tiled = nd.tile(nd.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])
#         # print('w_tiled',w_tiled.get_shape())
#         # w_tiled (100, 1152, 10, 8, 16)
#         # Transformed vectors, shape = [batch_size, input_num_capsule, num_capsule, 1, dim_vector]
#         inputs_hat = nd.batch_dot(inputs_tiled, w_tiled, [4, 3])
#         # print('inputs_hat',inputs_hat.get_shape())
#         # inputs_hat (100, 1152, 10, 1, 16)

#         # update self.bias by routing algorithm
#         for _ in range(self.num_routing):
#             c = nd.softmax(self.bias)
#             c_expand = nd.expand_dims(nd.expand_dims(nd.expand_dims(c, 2), 2), 0)
#             outputs = nd.sum(c_expand * inputs_hat, [1, 3], keepdims=True)
#             outputs = squash(outputs)
#             self.bias = self.bias + nd.sum(inputs_hat * outputs, [0, -2, -1])
#             # nd.update(self.bias, )
#         return nd.reshape(outputs, [self.batch_size, self.num_capsule, self.dim_vector])


class PrimaryCap(nn.Block):
    def __init__(self,dim_vector,n_channels,kernel_size,padding,strides=(1,1),**kwargs):
        super(PrimaryCap, self).__init__(**kwargs)
        self.squash = squash()
        self.net = nn.Sequential()
        self.output = null
        self.outputs = []
        self.s_squared_norm = 0
        self.scale = 0
        self.dim_vector = dim_vector
        self.n_channels=n_channels
        
        with self.name_scope():
            self.conv = nn.Conv2D(channels=n_channels,kernel_size=kernel_size,strides=strides,padding=padding,activation="relu")
            self.net.add(self.conv)

    def forward(self, x):
        for _ in range(self.n_channels):
            self.output = self.net(x)

            self.outputs.append(nd.reshape(data=self.output,shape=(self.output.shape[1] ** 2, self.dim_vector)))
            print(output.shape)
            print(outputs[-1].shape)
        
        self.outputs = nd.concatenate(outputs, axis=1)#¶(outputs)
        
        # squash
        self.s_squared_norm = nd.sum(nd.square(self.outputs), -1, keepdims=True)
        self.scale = self.s_squared_norm / (1 + self.s_squared_norm) / nd.sqrt(self.s_squared_norm)

        return self.scale * self.outputs





# class PrimaryCapt(nn.Block):
#     def __init__(self,dim_vector, n_channels, kernel_size, strides=(1,1), padding, **kwargs):
#         super(PrimaryCapt, self).__init__(**kwargs)
#         self.squash = squash()
#         self.net = nn.Sequential()
#         self.output = null
#         self.outputs = []
#         self.s_squared_norm = 0
#         self.scale = 0
#         self.dim_vector = dim_vector
#         self.n_channels=n_channels
#         with self.name_scope():
#             self.conv = nn.Conv2D(channels=n_channels,kernel_size=kernel_size,strides=strides,padding=padding,activation="relu")
#             self.net.add(self.conv)
#             # self.net.add(nn.Dense(128, activation="relu"))
#             # self.dense = nn.Dense(64)

#     def forward(self, x):
#         for _ in range(self.n_channels):
#             self.output = self.net(x)

#             self.outputs.append(nd.reshape(data=self.output,shape=(self.output.shape[1] ** 2, self.dim_vector)))
#             print(output.shape)
#             print(outputs[-1].shape)
        
#         self.outputs = nd.concatenate(outputs, axis=1)#¶(outputs)
        
#         # squash
#         self.s_squared_norm = nd.sum(nd.square(self.outputs), -1, keepdims=True)
#         self.scale = self.s_squared_norm / (1 + self.s_squared_norm) / nd.sqrt(self.s_squared_norm)

#         return self.scale * self.outputs
