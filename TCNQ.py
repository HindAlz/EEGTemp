import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import gc
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Activation, Permute, Dropout, Conv2D, AveragePooling2D,
    BatchNormalization, Input, Reshape, MultiHeadAttention, GlobalAveragePooling1D, DepthwiseConv2D, Conv1D, Add, LayerNormalization,Lambda
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import max_norm
from scipy.signal import welch


def clear_tf_memory():
    tf.keras.backend.clear_session()
    gc.collect()

def Temporal(in_chans, in_samples, num_classes):
    inputs = Input(shape=(1, in_chans, in_samples)) #shape: (none, 1, 60, 400)
    x = Permute((3, 2, 1))(inputs)  #shape: (none, 400, 60, 1)

    F1 = 8  #no. of filters
    kernLength = 64
    poolSize = 8
    dropout = 0.4

    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False)(x)  #kernel: (64,1)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((poolSize, 1))(x)
    x = Dropout(dropout)(x)
    #(none, 75, 60, 8)
    #(none, time, channels, filters)

    shape = tf.keras.backend.int_shape(x)  # (batch, time, channels, filters)
    time_steps = shape[1]
    channels = shape[2]
    filters = shape[3]

    x = Reshape((time_steps, channels * filters))(x)

    x = MultiHeadAttention(num_heads=2, key_dim=3)(x, x)
    x = GlobalAveragePooling1D()(x)

    #x = Dense(32, activation='relu')(x)
    #x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=x)


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


def Spatial(in_chans, in_samples, num_classes):
    poolSize = 8
    dropout = 0.4

    inputs = Input(shape=(1, in_chans, in_samples)) #(none, 1, 60, 400)
    x = Permute((2, 3, 1))(inputs) #(none, 60, 400, 1) (none, channels, samples, 1)

    x = DepthwiseConv2D(kernel_size=(in_chans, 1), depth_multiplier=2, use_bias=False, depthwise_constraint=max_norm(1.), padding="valid", data_format="channels_last")(x)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, poolSize))(x)
    x = Dropout(dropout)(x)
    # (none, 75, 60, 8)
    # (none, time, channels, filters)

    shape = tf.keras.backend.int_shape(x)  # (batch, time, channels, filters)
    time_steps = shape[1]
    channels = shape[2]
    filters = shape[3]

    x = Reshape((time_steps, channels * filters))(x)

    x = MultiHeadAttention(num_heads=2, key_dim=3)(x, x)
    x = GlobalAveragePooling1D()(x)

    return Model(inputs=inputs, outputs=x)

def Spectral(in_chans, in_samples, num_classes):
    inputs = Input(shape=(1, in_chans, in_samples)) #shape: (none, 1, channels, freq)
    x = Permute((3, 2, 1))(inputs)  #shape: (none, freq, channels, 1)

    F1 = 8
    kernLength = 64
    poolSize = 8
    dropout = 0.4

    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False)(x) #kernel across freq, channel is constant
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((poolSize, 1))(x)
    x = Dropout(dropout)(x)

    shape = tf.keras.backend.int_shape(x)  # (batch, time, channels, filters)
    time_steps = shape[1]
    channels = shape[2]
    filters = shape[3]

    x = Reshape((time_steps, channels * filters))(x)

    x = MultiHeadAttention(num_heads=2, key_dim=3)(x, x)
    x = GlobalAveragePooling1D()(x)

    return Model(inputs=inputs, outputs=x)


def psd(data, fs=200):
    chans = data.shape[0]
    trials = data.shape[2]
    all_psd = []

    for t in range(trials):
        trial_psd = []
        for c in range(chans):
            sig = data[c, :, t] #data is (channel, time, trial)
            freqs, power = welch(sig, fs=fs, nperseg=400, noverlap=200) #more and less
            trial_psd.append(power)
        all_psd.append(trial_psd)
    return np.array(all_psd)  #(trials, channels, freqs)

def threeBranch(input_shape_raw, input_shape_psd, num_classes,
                num_heads=2, key_dim=3, dense_size=128, act="relu"):
    inp_raw = Input(shape=input_shape_raw)
    inp_psd = Input(shape=input_shape_psd)
    regRate = .25
    reshaped_raw = Reshape((input_shape_raw[2], input_shape_raw[1]))(inp_raw)

    #learning rate?
    Q = TCN_block(input_layer=reshaped_raw, input_dimension=400, depth=2, kernel_size=1000, filters=32, dropout=0.3, activation='elu')
    K = Spatial(in_chans=input_shape_raw[1], in_samples=input_shape_raw[2], num_classes=num_classes)(inp_raw)
    V = Spectral(in_chans=input_shape_psd[1], in_samples=input_shape_psd[2], num_classes=num_classes)(inp_psd)

    Q = LayerNormalization()(Q)
    K = LayerNormalization()(K)
    V = LayerNormalization()(V)

    K = Lambda(lambda x: tf.expand_dims(x, axis=1))(K)
    V = Lambda(lambda x: tf.expand_dims(x, axis=1))(V)

    attention_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(Q, K, V)
    attention_out = GlobalAveragePooling1D()(attention_out)

    x = Dense(dense_size, activation=act)(attention_out)
    x = Dropout(0.4)(x)

    x = Dense(2, kernel_constraint=max_norm(regRate))(x)
    x = Activation('softmax', name='softmax')(x)

    return Model(inputs=[inp_raw, inp_psd], outputs=x)

def losoModel_one_subject_all_configs(X_data_list, Y_data_list, X_test_list, Y_test_list, subject_idx, num_classes, sfreq=200, avg=False):
    print("subject ", subject_idx)
    clear_tf_memory()

    #test data
    test_data = X_test_list[subject_idx].squeeze(1).transpose(1, 2, 0)
    X_test_psd = psd(test_data, fs=sfreq)
    X_test_psd = np.expand_dims(X_test_psd, axis=1)
    y_test = Y_test_list[subject_idx]
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    #training data
    X_train_raw = np.concatenate(X_data_list[:subject_idx] + X_data_list[subject_idx + 1:], axis=0)
    y_train = np.concatenate(Y_data_list[:subject_idx] + Y_data_list[subject_idx + 1:], axis=0)

    psd_train_list = []
    for i in range(len(X_data_list)):
        if i == subject_idx:
            continue
        d = X_data_list[i].squeeze(1).transpose(1, 2, 0)
        psd_d = psd(d, fs=sfreq)
        psd_train_list.append(np.expand_dims(psd_d, axis=1))
    X_train_psd = np.concatenate(psd_train_list, axis=0)

    y_train_cat = to_categorical(y_train, num_classes=num_classes)

    input_shape_raw = X_train_raw.shape[1:]
    input_shape_psd = X_train_psd.shape[1:]

    #configs
    num_heads = 7
    key_dim = 10
    Dsize = 256
    act = "relu"
    attn = ['Q', 'K', 'V']

    config_name = f"heads{num_heads}_dim{key_dim}_denseSize{Dsize}_activ{act},avg{avg},attn{attn}"

    print("\nconfig:", config_name)
    model = threeBranch(
        input_shape_raw=input_shape_raw,
        input_shape_psd=input_shape_psd,
        num_classes=num_classes,
        num_heads=num_heads,
        key_dim=key_dim,
        dense_size=Dsize,
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [EarlyStopping(patience=5, restore_best_weights=True)]

    model.fit([X_train_raw, X_train_psd], y_train_cat,
              epochs=30, batch_size=16, validation_split=0.2,
              callbacks=callback, verbose=1)
    import time
    start_time = time.time()
    loss, acc = model.evaluate([X_data_list[subject_idx], X_test_psd], y_test_cat, verbose=0)
    end_time = time.time()
    print("test time: ",end_time-start_time)
    print(f"accuracy: {acc:.4f}")

    return acc



with open("processed_data_mi_nonMi_Avg.pkl", 'rb') as f:
    X_data_list, Y_data_list = pickle.load(f)

with open("processed_data_mi_nonMi.pkl", 'rb') as f:
    X_data_list2, Y_data_list2 = pickle.load(f)

num_subjects = len(X_data_list)
num = 5

results = "results_avgInPrep.pkl"
if os.path.exists(results):
    with open(results, 'rb') as f:
        all_results = pickle.load(f)
else:
    all_results = {}

all_subject_accs = []

for subj in range(num):
    key = f"subject_{subj}"
    if key in all_results:
        print("already done")
        all_subject_accs.append(all_results[key]['accuracies'])
        continue

    acc = losoModel_one_subject_all_configs(
        X_data_list, Y_data_list, X_data_list2, Y_data_list2 , subject_idx=subj, num_classes=2
    )
    all_subject_accs.append(acc)
    all_results[key] = {
        'accuracies': acc,
    }

    with open(results, 'wb') as f:
        pickle.dump(all_results, f)

overall_avg = np.mean(all_subject_accs)
print("average accuracy",overall_avg)
