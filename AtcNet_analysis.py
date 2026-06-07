from tensorboard.plugins.hparams import api as hp
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D,
    AveragePooling2D, Add, SeparableConv2D, DepthwiseConv2D,
    BatchNormalization, SpatialDropout2D, Reshape, Input,
    Flatten, MultiHeadAttention, LayerNormalization, Conv1D,
    Concatenate, Lambda, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, GlobalAveragePooling1D
)
import json

import math
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Lambda, Dense
from tensorflow.keras.constraints import max_norm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.constraints import max_norm



def ATCNetDense(n_classes, in_chans=55, in_samples=1000):
    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])

    regRate = .25
    x = Flatten()(input_1)

    sw_concat = Dense(n_classes, kernel_constraint=max_norm(regRate))(x)
    softmax = Activation('softmax', name='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=softmax)

def ATCNetBlock1(n_classes, in_chans=55, in_samples=1000, n_windows=3, attention="mha",
           eegn_F1=16, eegn_D=2, eegn_kernelSize=1000, eegn_poolSize=8, eegn_dropout=0.1,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average'):

    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)  # (None, 1125, 22, 1) (none, seconds, chennels, 1)
    regRate = .25

    block1 = Conv_block(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                        kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                        in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)
    block1 = GlobalAveragePooling1D()(block1)
    sw_concat = Dense(n_classes, kernel_constraint=max_norm(regRate))(block1)
    softmax = Activation('softmax', name='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=softmax)



def ATCNetBlock2(n_classes, in_chans=55, in_samples=1000, n_windows=3, attention="mha",
           eegn_F1=16, eegn_D=2, eegn_kernelSize=1000, eegn_poolSize=8, eegn_dropout=0.1,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average'):
    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)  # (None, 1125, 22, 1) (none, seconds, chennels, 1)
    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    block1 = Conv_block(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                        kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                        in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    block2 = attention_block(block1, attention)

    block2 = GlobalAveragePooling1D()(block2)
    output = Dense(n_classes, kernel_constraint=max_norm(regRate))(block2)
    softmax = Activation('softmax', name='softmax')(output)

    return Model(inputs=input_1, outputs=softmax)


def ATCNetBlock3(n_classes, in_chans=55, in_samples=1000, n_windows=3, attention="mha",
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=1000, eegn_poolSize=8, eegn_dropout=0.1,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 tcn_activation='elu', fuse='average'):
    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)  # (None, 1125, 22, 1) (none, seconds, chennels, 1)
    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    block1 = Conv_block(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                        kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                        in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    block2 = attention_block(block1, attention)

    block3 = TCN_block(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                       kernel_size=tcn_kernelSize, filters=tcn_filters,
                       dropout=tcn_dropout, activation=tcn_activation)

    block3 = Lambda(lambda x: x[:, -1, :])(block3)

    output = Dense(n_classes, kernel_constraint=max_norm(regRate))(block3)
    softmax = Activation('softmax', name='softmax')(output)

    return Model(inputs=input_1, outputs=softmax)


def ATCNet(n_classes, in_chans=55, in_samples=1000, n_windows=3, attention="mha",
           eegn_F1=16, eegn_D=2, eegn_kernelSize=1000, eegn_poolSize=8, eegn_dropout=0.1,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
           tcn_activation='elu', fuse='average'):
    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, 22, 1125])
    input_2 = Permute((3, 2, 1))(input_1)  # (None, 1125, 22, 1) (none, seconds, chennels, 1)
    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    block1 = Conv_block(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                        kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                        in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)

    sw_concat = []

    for i in range(n_windows):
        st = i
        end = block1.shape[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]

        if attention is not None:
            block2 = attention_block(block2, attention)

        block3 = TCN_block(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                           kernel_size=tcn_kernelSize, filters=tcn_filters,
                           dropout=tcn_dropout, activation=tcn_activation)

        block3 = Lambda(lambda x: x[:, -1, :])(block3)


        if (fuse == 'average'):
            sw_concat.append(Dense(n_classes, kernel_constraint=max_norm(regRate))(block3))
        elif (fuse == 'concat'):
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if (fuse == 'average'):
        if len(sw_concat) > 1:  # more than one window
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:  # one window (# windows = 1)
            sw_concat = sw_concat[0]
    elif (fuse == 'concat'):
        sw_concat = Dense(n_classes, kernel_constraint=max_norm(regRate))(sw_concat)

    softmax = Activation('softmax', name='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=softmax)


def Conv_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
    """ Conv_block

        Notes
        -----
        This block is the same as EEGNet with SeparableConv2D replaced by Conv2D
        The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
        See details at https://arxiv.org/abs/1611.08024
    """
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, in_chans), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


def attention_block(net, attention_model):
    in_sh = net.shape  # dimensions of the input tensor
    in_len = len(in_sh)
    expanded_axis = 3  # defualt = 3

    if attention_model == 'mha':  # Multi-head self attention layer
        if (in_len > 3):
            net = Reshape((in_sh[1], -1))(net)
        net = mha_block(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_model))

    '''
    elif attention_model == 'mhla':  # Multi-head local self-attention layer
        if (in_len > 3):
            net = Reshape((in_sh[1], -1))(net)
        net = mha_block(net, vanilla=False)
    elif attention_model == 'se':  # Squeeze-and-excitation layer
        if (in_len < 4):
            net = tf.expand_dims(net, axis=expanded_axis)
        net = se_block(net, ratio=8)
    elif attention_model == 'cbam':  # Convolutional block attention module
        if (in_len < 4):
            net = tf.expand_dims(net, axis=expanded_axis)
        net = cbam_block(net, ratio=8)
    '''

    if (in_len == 3 and len(net.shape) == 4):
        net = K.squeeze(net, expanded_axis)
    elif (in_len == 4 and len(net.shape) == 3):
        net = Reshape((in_sh[1], in_sh[2], in_sh[3]))(net)
    return net


# %% Temporal convolutional (TC) block used in the ATCNet model
def TCN_block(input_layer, input_dimension, depth, kernel_size, filters, dropout, activation='relu'):
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

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out


def mha_block(input_feature, key_dim=8, num_heads=2, dropout=0.5, vanilla=True):
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
        x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x)

    else: #used here
        # Create a multi-head local self-attention layer as described in
        # 'Vision Transformer for Small-Size Datasets' https://arxiv.org/abs/2112.13492v1

        # Build the diagonal attention mask
        NUM_PATCHES = input_feature.shape[1]
        diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
        diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

        # Create a multi-head local self attention layer.
        x = MultiHeadAttention_LSA(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(
            x, x, attention_mask=diag_attn_mask)

    x = Dropout(0.3)(x)
    # Skip connection
    mha_feature = Add()([input_feature, x])

    return mha_feature


# %% Multi head self Attention (MHA) block: Locality Self Attention (LSA)
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


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
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


with open('processed_data_mi_nonMi.pkl', 'rb') as f:
    X_data, Y_data = pickle.load(f)

counter=0

'''

accs=[]
for i in range(5):
    print("SUBJECT", i)
    X_test = X_data[i]
    Y_test_curr = Y_data[i]
    X_train = np.concatenate(X_data[:i] + X_data[i+1:], axis=0)
    Y = np.concatenate(Y_data[:i] + Y_data[i+1:], axis=0)

    Y_train = to_categorical(Y, 2)

    Y_test = to_categorical(Y_test_curr, 2)
    model = ATCNetDense(2, in_chans=60, in_samples=400)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=60, batch_size=128, validation_split=0.15, verbose=0)

    model_probs = model.predict(X_test)
    ##model_pred = model_probs.argmax(axis=-1)
    _, test_acc = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
    accs.append(test_acc)
    print("Test Acc", round(test_acc,4))

print(f"model {counter} accuracies: {accs}, avg {np.mean(accs):.4f}")

counter+=1

accs = []
for i in range(5):
    print("SUBJECT", i)
    X_test = X_data[i]
    Y_test_curr = Y_data[i]
    X_train = np.concatenate(X_data[:i] + X_data[i + 1:], axis=0)
    Y = np.concatenate(Y_data[:i] + Y_data[i + 1:], axis=0)

    Y_train = to_categorical(Y, 2)

    Y_test = to_categorical(Y_test_curr, 2)

    model = ATCNetBlock1(2, in_chans=60, in_samples=400)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=60, batch_size=128, validation_split=0.15, verbose=0)
    model_probs = model.predict(X_test)
    #model_pred = model_probs.argmax(axis=-1)
    _, test_acc = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
    accs.append(test_acc)
    print("Test Acc", round(test_acc, 4))

print(f"model {counter} accuracies: {accs}, avg {np.mean(accs):.4f}")
results = {
    "model_name": counter,
    "accuracies": accs,
    "average_accuracy": float(np.mean(accs))  # convert to Python float
}
with open(f"atcnet_{counter}_results.json", "w") as f:
    json.dump(results, f, indent=4)

counter+=1
counter+=1


accs = []
for i in range(5):
    print("SUBJECT", i)
    X_test = X_data[i]
    Y_test_curr = Y_data[i]
    X_train = np.concatenate(X_data[:i] + X_data[i + 1:], axis=0)
    Y = np.concatenate(Y_data[:i] + Y_data[i + 1:], axis=0)

    Y_train = to_categorical(Y, 2)

    Y_test = to_categorical(Y_test_curr, 2)

    model = ATCNetBlock2(2, in_chans=60, in_samples=400)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=60, batch_size=128, validation_split=0.15, verbose=0)
    model_probs = model.predict(X_test)
    #model_pred = model_probs.argmax(axis=-1)
    _, test_acc = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
    accs.append(test_acc)
    print("Test Acc", round(test_acc, 4))

print(f"model {counter} accuracies: {accs}, avg {np.mean(accs):.4f}")
results = {
    "model_name": counter,
    "accuracies": accs,
    "average_accuracy": float(np.mean(accs))  # convert to Python float
}
with open(f"atcnet_{counter}_results.json", "w") as f:
    json.dump(results, f, indent=4)

counter+=1

'''

accs = []
for i in range(5):
    print("SUBJECT", i)
    X_test = X_data[i]
    Y_test_curr = Y_data[i]
    X_train = np.concatenate(X_data[:i] + X_data[i + 1:], axis=0)
    Y = np.concatenate(Y_data[:i] + Y_data[i + 1:], axis=0)

    Y_train = to_categorical(Y, 2)

    Y_test = to_categorical(Y_test_curr, 2)
    model = ATCNetBlock3(2, in_chans=60, in_samples=400)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=60, batch_size=128, validation_split=0.15, verbose=0)
    model_probs = model.predict(X_test)
    #model_pred = model_probs.argmax(axis=-1)
    _, test_acc = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
    accs.append(test_acc)
    print("Test Acc", round(test_acc, 4))

print(f"model {counter} accuracies: {accs}, avg {np.mean(accs):.4f}")
results = {
    "model_name": counter,
    "accuracies": accs,
    "average_accuracy": float(np.mean(accs))  # convert to Python float
}
with open(f"atcnet_{counter}_results.json", "w") as f:
    json.dump(results, f, indent=4)

counter+=1
'''


accs = []
for i in range(5):
    print("SUBJECT", i)
    X_test = X_data[i]
    Y_test_curr = Y_data[i]
    X_train = np.concatenate(X_data[:i] + X_data[i + 1:], axis=0)
    Y = np.concatenate(Y_data[:i] + Y_data[i + 1:], axis=0)

    Y_train = to_categorical(Y, 2)

    Y_test = to_categorical(Y_test_curr, 2)

    model = ATCNet(2, in_chans=60, in_samples=400)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=60, batch_size=128, validation_split=0.15, verbose=0)
    model_probs = model.predict(X_test)
    #model_pred = model_probs.argmax(axis=-1)
    _, test_acc = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
    accs.append(test_acc)
    print("Test Acc", round(test_acc, 4))

print(f"model {counter}  accuracies: {accs}, avg {np.mean(accs):.4f}")
results = {
    "model_name": counter,
    "accuracies": accs,
    "average_accuracy": float(np.mean(accs))  # convert to Python float
}
with open(f"atcnet_{counter}_results.json", "w") as f:
    json.dump(results, f, indent=4)
'''

#dense  0.5469
#block1 0.6511
#block2
#block3
#atc 0.6829