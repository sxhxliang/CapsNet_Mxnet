from mxnet import nd
from mxnet.gluon import nn,Parameter
from mxnet import init
from mxnet import cpu
from mxnet.gluon.loss import Loss, L2Loss,  _apply_weighting





class PrimaryConv(nn.Block):
    def __init__(self,dim_vector,n_channels,kernel_size,padding,context=cpu,strides=(1,1),**kwargs):
        super(PrimaryConv, self).__init__(**kwargs)
    

        self.dim_vector = dim_vector
        self.n_channels= n_channels
        # self.conv_vector = nn.Conv2D(channels=dim_vector, kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')
       
        # self.caps = [self.conv_vector for x in range(self.n_channels)]

        self.batch_size = 0

        self.capsules_index = ['dim_'+str(i) for i in range(n_channels)]

        for idx in self.capsules_index:
            setattr(self, idx, nn.Conv2D(channels=dim_vector, 
                    kernel_size=kernel_size, strides=strides,
                    padding=padding,activation='relu'))


    def reshape_conv(self,conv_vector):
        return nd.reshape(conv_vector,shape=(self.batch_size, self.dim_vector,-1))


    def concat_outputs(self,conv_list,axis):

        concat_vec = conv_list[0]
        # print(concat_vec.shape)
        concat_vec = self.reshape_conv(concat_vec)
        for i in range(1, len(conv_list)):
            concat_vec = nd.concat(concat_vec, self.reshape_conv(conv_list[i]), dim=axis)
        
        return concat_vec

    def squash(self,vectors,axis):
        epsilon = 1e-9
        vectors_l2norm = nd.square(vectors).sum(axis=axis,keepdims=True)#.expand_dims(axis=axis)
    
        scale_factor = vectors_l2norm / (1 + vectors_l2norm) 
        vectors_squashed = scale_factor * (vectors / nd.sqrt(vectors_l2norm+epsilon)) # element-wise

        return vectors_squashed
   
    def forward(self, x):

        self.batch_size = x.shape[0]

        conv_list = [getattr(self,idx)(x).expand_dims(axis=-1) for idx in self.capsules_index]

        # conv_list = [ self.conv_vector(x) for i in range(self.n_channels)]
        
        outputs = self.concat_outputs(conv_list,axis=2)
        assert outputs.shape == (self.batch_size, 8, 1152)

        v_primary = self.squash(outputs,axis=1)
        assert outputs.shape == (self.batch_size, 8, 1152)

        return outputs


class DigitCaps(nn.Block):
    def __init__(self,num_capsule,dim_vector,context=cpu,iter_routing=1,**kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule #10
        self.dim_vector = dim_vector #16
        
        self.iter_routing = iter_routing #3

        self.batch_size = 1 
        self.input_num_capsule = 1152
        self.input_dim_vector = 8
        self.context = context

        self.routing_weight_initial = True

        if self.routing_weight_initial:
            self.routing_weight = nd.random_normal(shape=(
                1,
                self.input_num_capsule,
                self.num_capsule,
                self.input_dim_vector,
                self.dim_vector), name='routing_weight').as_in_context(self.context)
            self.routing_weight_initial = False
        
        self.routing_weight.attach_grad()

        #  (1, 1152, 10, 8, 16)
        self.W_ij  = self.params.get(
            'weight',shape=(
                1,
                self.input_num_capsule,
                self.num_capsule,
                self.input_dim_vector,
                self.dim_vector
                )) 

    def squash(self,vectors,axis):
        epsilon = 1e-9
        vectors_l2norm = nd.square(vectors).sum(axis=axis,keepdims=True)
        
        assert vectors_l2norm.shape == (self.batch_size, 1, self.num_capsule, 1, 1) # 1,10,1,1

        scale_factor = vectors_l2norm / (1 + vectors_l2norm) 
        vectors_squashed = scale_factor * (vectors / nd.sqrt(vectors_l2norm+epsilon)) # element-wise

        return vectors_squashed

  
    def forward(self, x):

        self.batch_size, self.input_dim_vector, self.input_num_capsule = x.shape

        assert (self.batch_size, self.input_dim_vector, self.input_num_capsule) == (self.batch_size,8,1152)

        x_exp = x.expand_dims(axis=1)
        x_exp = x_exp.expand_dims(axis=4)
        assert x_exp.shape == (self.batch_size, 1, 8, 1152, 1)


        x_tile = x_exp.tile(reps=[1, self.num_capsule, 1, 1, 1])
        assert x_tile.shape == (self.batch_size,10, 8, 1152, 1)


        x_trans = x_tile.transpose(axes=(0,3,1,2,4))
        assert x_trans.shape == (self.batch_size, 1152, 10, 8,1)

        # W = self.W_ij.data()
        print(self.W_ij.data()[0,0,0,0])
        # W = self.routing_weight
        # print('W',W[0,0,0,0])

        W = self.W_ij.data().tile(reps=[self.batch_size,1,1,1,1])

        assert W.shape == (self.batch_size, 1152, 10, 8, 16)


        # [8, 16].T x [8, 1] => [16, 1]
        x_dot = x_trans.reshape(shape=(-1,self.input_dim_vector,1))#(8,1)
        W_dot = W.reshape(shape=(-1,self.input_dim_vector,self.dim_vector))#(8,16)
       

        u_hat = nd.batch_dot(W_dot,x_dot,transpose_a=True)

        u_hat = u_hat.reshape(shape=(self.batch_size,self.input_num_capsule,self.num_capsule,self.dim_vector,-1))
        assert u_hat.shape == (self.batch_size, 1152, 10, 16, 1)

        
        b_IJ = nd.zeros((self.batch_size, self.input_num_capsule,self.num_capsule,1,1),ctx=self.context)

        assert b_IJ.shape == ((self.batch_size,1152,10,1,1))
        
        u_hat_stopped = nd.stop_gradient(u_hat, name='stop_gradient')


        for r_iter in range(self.iter_routing):
            c_IJ = nd.softmax(b_IJ, axis=2)

            s_J = nd.multiply(c_IJ, u_hat)
            s_J = s_J.sum(axis=1,keepdims=True)
            # print('s_J',s_J[0,0,0])

            assert s_J.shape == (self.batch_size, 1, 10, 16, 1)

            v_J = self.squash(s_J,axis=3)

            assert v_J.shape == (self.batch_size, 1, 10, 16, 1)

            v_J_tiled = v_J.tile(reps=[1, 1152, 1, 1, 1])

            if self.iter_routing > 1:
                # u_hat_stopped  (self.batch_size, 1152, 10, 16, 1)
                # v_J_tiled (self.batch_size, 1152, 10, 16, 1)
                # u_hat_stopped = u_hat_stopped.reshape(shape=(-1,self.dim_vector,1))
                # v_J_tiled = v_J_tiled.reshape(shape=(-1,self.dim_vector,1))
                # 
                u_produce_v = nd.stop_gradient(nd.multiply(u_hat_stopped, v_J_tiled, transpose_a=True))
               
                # u_produce_v = u_produce_v.reshape(shape=(self.batch_size, self.input_num_capsule, self.num_capsule, 1, 1))
                assert u_produce_v.shape == (self.batch_size, 1152, 10, 1, 1)
                
                b_IJ = nd.stop_gradient(b_IJ+u_produce_v, name ="update_b_IJ" )

        #(batch_size,1,10,16,1)
        assert v_J.shape == (self.batch_size,1,self.num_capsule,self.dim_vector,1)
        # print('v_J',v_J[0,0,0,0])
        return v_J
       


class Length(nn.Block):
    def __init__(self, **kwargs):
        super(Length, self).__init__(**kwargs)

    def forward(self, x):
        #(batch_size, 1, 10, 16, 1) =>(batch_size,10, 16)=> (batch_size, 10, 1)
        x_shape = x.shape
        x = x.reshape(shape=(x_shape[0],x_shape[2],x_shape[3]))

        x_l2norm = nd.sqrt((x.square()).sum(axis=-1))
        # prob = nd.softmax(x_l2norm, axis=-1)
        return x_l2norm

class CapsuleMarginLoss(Loss):
    """Calculates margin loss for CapsuleNet between output and label:

    .. math::
        

    Output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=1., batch_axis=0, num_classes=10, sample_weight=None, **kwargs):
        super(CapsuleMarginLoss, self).__init__(weight, batch_axis, **kwargs)
        self.lambda_value = 0.5
        self.num_classes = num_classes
        self.sample_weight = sample_weight

    def forward(self,labels,y_pred):
       
        labels_onehot = labels #nd.one_hot(labels, self.num_classes)
  

        first_term_base = nd.square(nd.maximum(0.9-y_pred,0))
        second_term_base = nd.square(nd.maximum(y_pred -0.1, 0))
        # import pdb; pdb.set_trace()
        margin_loss = labels_onehot * first_term_base + self.lambda_value * (1-labels_onehot) * second_term_base
        margin_loss = margin_loss.sum(axis=1) 

        loss = nd.mean(margin_loss, axis=self._batch_axis, exclude=True) 
        loss = _apply_weighting(nd, loss, self._weight/2, self.sample_weight)
        return nd.mean(loss, axis=self._batch_axis, exclude=True)
    

