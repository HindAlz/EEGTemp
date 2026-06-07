from tensorboard.plugins.hparams import api as hp
import numpy as np
import mat73
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D, 
    AveragePooling2D, Add, SeparableConv2D, DepthwiseConv2D, 
    BatchNormalization, SpatialDropout2D, Reshape, Input, 
    Flatten, MultiHeadAttention, LayerNormalization, Conv1D, 
    Concatenate, Lambda, GlobalAveragePooling2D, multiply
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import shap
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import keras
import time
from sklearn.model_selection import train_test_split

def mha_block(input_feature, key_dim=8, num_heads=2, dropout = 0.5, vanilla = True):
    
    """Multi Head self Attention (MHA) block.     
       
    Here we include two types of MHA blocks: 
            The original multi-head self-attention as described in https://arxiv.org/abs/1706.03762
            The multi-head local self attention as described in https://arxiv.org/abs/2112.13492v1
    """    
    # Layer normalization
    x = LayerNormalization(epsilon=1e-6)(input_feature)
    
    if vanilla:
        # Create a multi-head attention layer as described in 
        # 'Attention Is All You Need' https://arxiv.org/abs/1706.03762
        x = MultiHeadAttention(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(x, x)
        
    else:
        # Create a multi-head local self-attention layer as described in 
        # 'Vision Transformer for Small-Size Datasets' https://arxiv.org/abs/2112.13492v1
        
        # Build the diagonal attention mask
        NUM_PATCHES = input_feature.shape[1]
        diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
        diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
        
        # Create a multi-head local self attention layer.
        x = MultiHeadAttention_LSA(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(
            x, x, attention_mask = diag_attn_mask)
        
    x = Dropout(0.3)(x)
    # Skip connection
    mha_feature = Add()([input_feature, x])
    
    return mha_feature


#%% Multi head self Attention (MHA) block: Locality Self Attention (LSA)
class MultiHeadAttention_LSA(tf.keras.layers.MultiHeadAttention):
    """local multi-head self attention block
     
     Locality Self Attention as described in https://arxiv.org/abs/2112.13492v1
     This implementation is taken from  https://keras.io/examples/vision/vit_small_ds/ 
    """    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is the square 
        # root of the key dimension.
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


#%% Squeeze-and-excitation block
def se_block(input_feature, ratio=8):
    """Squeeze-and-Excitation(SE) block.
    
    As described in https://arxiv.org/abs/1709.01507
    The implementation is taken from https://github.com/kobiso/CBAM-keras
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
                        activation='relu',
                       kernel_initializer='he_normal',
                      use_bias=True,
                     bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
            se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


#%% Convolutional block attention module
def cbam_block(cbam_feature, ratio=8):
    """ Convolutional Block Attention Module(CBAM) block.
    
    As described in https://arxiv.org/abs/1807.06521
        The implementation is taken from https://github.com/kobiso/CBAM-keras
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
# channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert cbam_feature.shape[-1] == 1
        
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

#%% Create and apply the attention model
def attention_block(net, attention_model): 
    in_sh = net.shape # dimensions of the input tensor
    in_len = len(in_sh) 
    expanded_axis = 3 # defualt = 3
    
    if attention_model == 'mha':   # Multi-head self attention layer 
        if(in_len > 3):
            net = Reshape((in_sh[1],-1))(net)
        net = mha_block(net)
    elif attention_model == 'mhla':  # Multi-head local self-attention layer 
        if(in_len > 3):
            net = Reshape((in_sh[1],-1))(net)
        net = mha_block(net, vanilla = False)
    elif attention_model == 'se':   # Squeeze-and-excitation layer
        if(in_len < 4):
            net = tf.expand_dims(net, axis=expanded_axis)
        net = se_block(net, ratio=8)
    elif attention_model == 'cbam': # Convolutional block attention module
        if(in_len < 4):
            net = tf.expand_dims(net, axis=expanded_axis)
        net = cbam_block(net, ratio=8)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_model))
        
    if (in_len == 3 and len(net.shape) == 4):
        net = K.squeeze(net, expanded_axis)
    elif (in_len == 4 and len(net.shape) == 3):
        net = Reshape((in_sh[1], in_sh[2], in_sh[3]))(net)
    return net
        
#%% The proposed ATCNet model, https://doi.org/10.1109/TII.2022.3197419
def ATCNet(n_classes,in_chans = 55, in_samples = 1000, n_windows = 9, attention = "mha", 
           eegn_F1 = 16, eegn_D = 2, eegn_kernelSize = 1000, eegn_poolSize = 8, eegn_dropout=0.1, 
           tcn_depth = 2, tcn_kernelSize = 4, tcn_filters = 32, tcn_dropout = 0.3, 
           tcn_activation = 'elu', fuse = 'average'):
    """ ATCNet model from Altaheri et al 2022.
        See details at https://ieeexplore.ieee.org/abstract/document/9852687
    
        Notes
        -----
        The initial values in this model are based on the values identified by
        the authors
        
        References
        ----------
        .. H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed 
           attention temporal convolutional network for EEG-based motor imagery 
           classification," in IEEE Transactions on Industrial Informatics, 2022, 
           doi: 10.1109/TII.2022.3197419.
    """
    input_1 = Input(shape = (1,in_chans, in_samples))   #     TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3,2,1))(input_1)  #(None, 1125, 22, 1) (none, seconds, chennels, 1)
    regRate=.25
    numFilters = eegn_F1
    F2 = numFilters*eegn_D

    block1 = Conv_block(input_layer = input_2, F1 = eegn_F1, D = eegn_D, 
                        kernLength = eegn_kernelSize, poolSize = eegn_poolSize,
                        in_chans = in_chans, dropout = eegn_dropout)
    block1 = Lambda(lambda x: x[:,:,-1,:])(block1)
    
    
   
     
    # Sliding window 
    sw_concat = []   # to store concatenated or averaged sliding window outputs
    
    for i in range(n_windows):
        st = i
        end = block1.shape[1]-n_windows+i+1
        block2 = block1[:, st:end, :]
        
        # Attention_model
        if attention is not None:
            block2 = attention_block(block2, attention)

        # Temporal convolutional network (TCN)
        block3 = TCN_block(input_layer = block2, input_dimension = F2, depth = tcn_depth,
                            kernel_size = tcn_kernelSize, filters = tcn_filters, 
                            dropout = tcn_dropout, activation = tcn_activation)
        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:,-1,:])(block3)
        
        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if(fuse == 'average'):
            sw_concat.append(Dense(n_classes, kernel_constraint = max_norm(regRate))(block3))
        elif(fuse == 'concat'):
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])
                
    if(fuse == 'average'):
        if len(sw_concat) > 1: # more than one window
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else: # one window (# windows = 1)
            sw_concat = sw_concat[0]
    elif(fuse == 'concat'):
        sw_concat = Dense(n_classes, kernel_constraint = max_norm(regRate))(sw_concat)
            
    
    softmax = Activation('softmax', name = 'softmax')(sw_concat)
    
    return Model(inputs = input_1, outputs = softmax)

#%% Convolutional (CV) block used in the ATCNet model
def Conv_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
    """ Conv_block
    
        Notes
        -----
        This block is the same as EEGNet with SeparableConv2D replaced by Conv2D 
        The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
        See details at https://arxiv.org/abs/1611.08024
    """
    F2 = F1*D
    block1 = Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)
    block2 = DepthwiseConv2D((1, in_chans), use_bias = False, 
                                    depth_multiplier = D,
                                    data_format='channels_last',
                                    depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((poolSize,1),data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                            data_format='channels_last',
                            use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    
    block3 = AveragePooling2D((poolSize,1),data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3

#%% Temporal convolutional (TC) block used in the ATCNet model
def TCN_block(input_layer,input_dimension,depth,kernel_size,filters,dropout,activation='relu'):
    """ TCN_block from Bai et al 2018
        Temporal Convolutional Network (TCN)
        
        Notes
        -----
        THe original code available at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
        This implementation has a slight modification from the original code
        and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
        See details at https://arxiv.org/abs/2006.00622

        References
        ----------
        .. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
           An empirical evaluation of generic convolutional and recurrent networks
           for sequence modeling.
           arXiv preprint arXiv:1803.01271.
    """    
    
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth-1):
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
        
    return out


def proposed(n_timesteps, n_features, n_outputs, f):
    

   
    input_1 = Input(shape=(1, n_features, n_timesteps))  # TensorShape([None, 1, 22, 1125])

    # block0       = Conv2D(filters=8, kernel_size=(1, 16), use_bias = False, padding='same', data_format="channels_first")(input_1)
    # block0       = LayerNormalization()(block0)
    # block0       = Activation(activation='elu')(block0)

    block1       = Conv2D(filters=f, kernel_size=(1, 32), use_bias = False, padding='same', data_format="channels_first")(input_1)
    block1       = LayerNormalization()(block1)
    block1       = Activation(activation='elu')(block1)
    
  
    block2       = Conv2D(filters=f, kernel_size=(1, 64), use_bias = False, padding='same', data_format="channels_first")(input_1)
    block2       = LayerNormalization()(block2)
    block2       = Activation(activation='elu')(block2)

    # block2x       = Conv2D(filters=8, kernel_size=(1, 8), use_bias = False, padding='same', data_format="channels_first")(input_1)
    # block2x       = LayerNormalization()(block2x)
    # block2x       = Activation(activation='elu')(block2x)

    # block2xx       = Conv2D(filters=8, kernel_size=(1, 4), use_bias = False, padding='same', data_format="channels_first")(input_1)
    # block2xx       = LayerNormalization()(block2xx)
    # block2xx       = Activation(activation='elu')(block2xx)


    block2 = Concatenate(axis=1)([  block1, block2])
    block2       = se_block(block2, 8)
    
    
    
    block3       = DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias = False, depthwise_constraint=max_norm(1.), data_format="channels_first")(block2)
    block3       = LayerNormalization()(block3)
    block3       = Activation(activation='elu')(block3)

    block3       = AveragePooling2D(pool_size=(1, 64), padding='same', data_format="channels_first")(block3)

    block5  = Flatten() (block3)
    block5       = Dense(n_outputs, kernel_constraint=max_norm(0.25))(block5)
    block5       = Activation(activation='softmax')(block5)

    return Model(inputs=input_1, outputs=block5)


def se_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 #if K.image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D(data_format="channels_first")(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    #if K.image_data_format() == 'channels_first':
    se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x



def attention_block_Irvan(inputs):
    attention_probs = Dense(inputs.shape[-1], activation='softmax')(inputs)
    attention_mul = tf.keras.layers.multiply([inputs, attention_probs])
    return attention_mul

# Transformer encoder block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

# Function to define EEGNeX + Transformer model
def EEGNeX_Transformer(n_timesteps, n_features, n_outputs):
    input_main = Input(shape=(1, n_features, n_timesteps)) 

    block1 = Conv2D(filters=8, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first")(input_main)
    block1 = LayerNormalization()(block1)
    block1 = Activation(activation='elu')(block1)

    block2 = Conv2D(filters=32, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first")(block1)
    block2 = LayerNormalization()(block2)
    block2 = Activation(activation='elu')(block2)

    block3 = DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias=False, depthwise_constraint=max_norm(1.), data_format="channels_first")(block2)
    block3 = LayerNormalization()(block3)
    block3 = Activation(activation='elu')(block3)
    block3 = AveragePooling2D(pool_size=(1, 4), padding='same', data_format="channels_first")(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(filters=32, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 2), data_format='channels_first')(block3)
    block4 = LayerNormalization()(block4)
    block4 = Activation(activation='elu')(block4)

    block5 = Conv2D(filters=8, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 4), data_format='channels_first')(block4)
    block5 = LayerNormalization()(block5)
    block5 = Activation(activation='elu')(block5)
    block5 = Dropout(0.5)(block5)
    block5 = Lambda(lambda x: x[:,:,-1,:])(block5)

    # Adding Transformer Encoder layers
    block5 = Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0, 2, 1)))(block5)  # Permute dimensions to fit Transformer input format
    block5_shape = K.int_shape(block5)
    block5 = Reshape((block5_shape[1], block5_shape[2]))(block5)
    block5 = transformer_encoder(block5, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    block5 = transformer_encoder(block5, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    block5 = Flatten()(block5)
    
    # Adding Attention mechanism
    block5 = attention_block_Irvan(block5)
    
    block5 = Dense(128, activation='relu')(block5)
    block5 = Dropout(0.5)(block5)
    block5 = Dense(n_outputs, kernel_constraint=max_norm(0.25))(block5)
    block5 = Activation(activation='softmax')(block5)
    return Model(inputs=input_main, outputs=block5)



def EEGNeX_8_32(n_timesteps, n_features, n_outputs):

    # start the model
    input_main = Input(shape=(1, n_features, n_timesteps)) 
    block1 = Conv2D(filters=8, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first")(
        input_main)
    block1 = LayerNormalization()(block1)
    block1 = Activation(activation='elu')(block1)

    block2 = Conv2D(filters=32, kernel_size=(1, 32), use_bias=False, padding='same', data_format="channels_first")(
        block1)
    block2 = LayerNormalization()(block2)
    block2 = Activation(activation='elu')(block2)

    block3 = DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, use_bias=False,
                             depthwise_constraint=max_norm(1.), data_format="channels_first")(block2)
    block3 = LayerNormalization()(block3)
    block3 = Activation(activation='elu')(block3)
    block3 = AveragePooling2D(pool_size=(1, 4), padding='same', data_format="channels_first")(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(filters=32, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 2),
                    data_format='channels_first')(block3)
    block4 = LayerNormalization()(block4)
    block4 = Activation(activation='elu')(block4)

    block5 = Conv2D(filters=8, kernel_size=(1, 16), use_bias=False, padding='same', dilation_rate=(1, 4),
                    data_format='channels_first')(block4)
    block5 = LayerNormalization()(block5)
    block5 = Activation(activation='elu')(block5)
    block5 = Dropout(0.5)(block5)

    block5 = Flatten()(block5)
    block5 = Dense(n_outputs, kernel_constraint=max_norm(0.25))(block5)
    block5 = Activation(activation='softmax')(block5)
    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='EEGNeX_8_32.png')
    return Model(inputs=input_main, outputs=block5)



#%% need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))  

#%% Reproduced ShallowConvNet model: https://doi.org/10.1002/hbm.23730
def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    See details at https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
       
    The original code for this model is available at: https://github.com/braindecode/braindecode

        Notes
        -----
        The initial values in this model are based on the values identified by the authors

        This implementation is taken from code by the Army Research Laboratory (ARL) 
        at https://github.com/vlawhern/arl-eegmodels
       
        References
        ----------
        .. Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
           Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). 
           Deep learning with convolutional neural networks for EEG decoding 
           and visualization. Human brain mapping, 38(11), 5391-5420.

    """
    # start the model
    # input_main   = Input((Chans, Samples, 1))
    input_main   = Input((1, Chans, Samples))
    input_2 = Permute((2,3,1))(input_main) 

    block1       = Conv2D(40, (1, 25), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_2)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)