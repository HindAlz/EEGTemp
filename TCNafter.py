import mne
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from tensorflow.keras import (
    Model, layers
)
from tensorflow.keras.layers import (
    Dense, Activation, Permute, Dropout, Conv2D, AveragePooling2D,
    BatchNormalization, Input, Reshape, MultiHeadAttention, GlobalAveragePooling1D, GlobalAveragePooling2D, DepthwiseConv2D, Conv1D, Add
)
from tensorflow.keras.constraints import max_norm
from scipy.signal import welch
from mne.io import BaseRaw
from warnings import warn
from mne import BaseEpochs, create_info
import scipy

def clear_tf_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def Temporal(in_chans, in_samples, num_classes,
             tcn_depth=2, tcn_kernel_size=64, tcn_filters=8, tcn_dropout=0.4, tcn_activation='elu',
             attn_num_heads=2, attn_key_dim=3):
    inputs = Input(shape=(1, in_chans, in_samples)) #shape: (none, 1, 60, 400)
    x = Permute((3, 2, 1))(inputs)  #shape: (none, 400, 60, 1)

    F1 = 8  #no. of filters
    kernLength = 64
    poolSize = 8
    dropout = 0.4

    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False)(x)  #kernel: (64,1) , test with depthwise?
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
    x = TCN_block(x, depth=tcn_depth, kernel_size=tcn_kernel_size,
                  filters=tcn_filters * channels, dropout=tcn_dropout, activation=tcn_activation, input_dimension=x.shape[-1])

    x = MultiHeadAttention(num_heads=attn_num_heads, key_dim=attn_key_dim)(x, x)
    x = GlobalAveragePooling1D()(x)

    #x = Dense(32, activation='relu')(x)
    #outputs = Dense(num_classes, activation='softmax')(x)

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

    x = DepthwiseConv2D(kernel_size=(in_chans, 1), depth_multiplier=2, use_bias=False,
                        depthwise_constraint=max_norm(1.), padding="valid", data_format="channels_last")(x)

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
    #x = Dense(32, activation='relu')(x)
    #print(x.shape)
    #outputs = Dense(num_classes, activation='softmax')(x)

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

    #x = Dense(32, activation='relu')(x)
    #outputs = Dense(num_classes, activation='softmax')(x)
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
from tensorflow.keras.layers import Lambda

from tensorflow.keras.layers import LayerNormalization


def threeBranch(input_shape_raw, input_shape_psd, num_classes,
                add_dense=False, add_layernorm=False,
                num_heads=2, key_dim=3, dense_size=128, act="relu"):
    inp_raw = Input(shape=input_shape_raw)
    inp_psd = Input(shape=input_shape_psd)

    Q = Temporal(in_chans=input_shape_raw[1], in_samples=input_shape_raw[2], num_classes=num_classes)(inp_raw)
    K = Spatial(in_chans=input_shape_raw[1], in_samples=input_shape_raw[2], num_classes=num_classes)(inp_raw)
    V = Spectral(in_chans=input_shape_psd[1], in_samples=input_shape_psd[2], num_classes=num_classes)(inp_psd)

    from tensorflow.keras.layers import Lambda
    Q = Lambda(lambda x: tf.expand_dims(x, axis=1))(Q)
    K = Lambda(lambda x: tf.expand_dims(x, axis=1))(K)
    V = Lambda(lambda x: tf.expand_dims(x, axis=1))(V)

    if add_layernorm:
        Q = LayerNormalization()(Q)
        K = LayerNormalization()(K)
        V = LayerNormalization()(V)

    attention_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(Q, K, V)
    regRate = .25
    F2 = attention_out.shape[-1]  # or use tf.shape if dynamic
    block3 = TCN_block(input_layer=attention_out, input_dimension=F2, depth=2,
                       kernel_size=1000, filters=32, dropout=0.3, activation='elu')

    # Get feature maps of the last sequence
    block3 = Lambda(lambda x: x[:, -1, :])(block3)
    x = Dense(dense_size, activation=act)(block3)
    x = Dropout(0.4)(x)
    output = Dense(2, kernel_constraint=max_norm(regRate))(x)
    softmax = Activation('softmax', name='softmax')(output)

    return Model(inputs=[inp_raw, inp_psd], outputs=softmax)
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping
import itertools
import hashlib


def losoModel_one_subject_all_configs(X_data_list, Y_data_list, X_test_list, Y_test_list, subject_idx, cond1, cond2, num_classes, sfreq=200,
                                      avg=False):
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

    #config
    dense_options = [True]
    layernorm_options = [True]
    earlystop_options = [True]
    num_heads_options = [7]
    key_dim_options = [10]
    dense_sizes = [256]
    activations = ["elu"]
    attns = [['Q', 'K', 'V']]


    all_accuracies = []

    for add_dense in dense_options:
        for add_layernorm in layernorm_options:
            for use_earlystop in earlystop_options:
                for num_heads in num_heads_options:
                    for key_dim in key_dim_options:
                        for Dsize in dense_sizes:
                            for act in activations:
                                for attn in attns:
                                    config_name = f"heads{num_heads}_dim{key_dim}_denseSize{Dsize}_activ{act}, _avg{avg}, _attn{attn}_TCN_merge_Temporal_filefix_Avg"
                                    results_file = f"5loso_subject{subject_idx}_{config_name}.pkl"

                                    print(f"\n[CONFIG] {config_name}")

                                    if os.path.exists(results_file):
                                        print("→ Skipping (already done)")
                                        with open(results_file, 'rb') as f:
                                            saved = pickle.load(f)
                                            acc = saved['accuracy']
                                            all_accuracies.append(acc)
                                        continue

                                    model = threeBranch(
                                        input_shape_raw=input_shape_raw,
                                        input_shape_psd=input_shape_psd,
                                        num_classes=num_classes,
                                        add_dense=add_dense,
                                        add_layernorm=add_layernorm,
                                        num_heads=num_heads,
                                        key_dim=key_dim,
                                        dense_size=Dsize,
                                    )

                                    model.compile(optimizer='adam', loss='categorical_crossentropy',
                                                  metrics=['accuracy'])

                                    callbacks = [EarlyStopping(monitor='val_loss', patience=5,
                                                               restore_best_weights=True)] if use_earlystop else []

                                    model.fit([X_train_raw, X_train_psd], y_train_cat,
                                              epochs=30, batch_size=16, validation_split=0.2,
                                              callbacks=callbacks, verbose=0)

                                    loss, acc = model.evaluate([X_data_list[subject_idx], X_test_psd], y_test_cat,
                                                               verbose=0)
                                    print(f"→ Accuracy: {acc:.4f}")

                                    all_accuracies.append(acc)

                                    with open(results_file, 'wb') as f:
                                        pickle.dump({
                                            'config': {
                                                'dense': add_dense,
                                                'layernorm': add_layernorm,
                                                'earlystop': use_earlystop,
                                                'num_heads': num_heads,
                                                'key_dim': key_dim,
                                            },
                                            'accuracy': acc
                                        }, f)


    #summary
    avg_acc = np.mean(all_accuracies)
    print(f"\n finished configs. avg accuracy: {avg_acc:.4f}")
    return all_accuracies, avg_acc


with open("processed_data_mi_nonMi.pkl", 'rb') as f:
    X_data_list, Y_data_list = pickle.load(f)
with open("processed_data_mi_nonMi_Avg.pkl", 'rb') as f:
    X_data_list2, Y_data_list2 = pickle.load(f)

num_subjects = len(X_data_list)
half = 5
all_subject_avg_accs = []

for subj in range(half):
    print(f"\n==== Running LOSO Config Test for Subject {subj}/{half - 1} ====")
    accs, avg = losoModel_one_subject_all_configs(
        X_data_list, Y_data_list, X_data_list2, Y_data_list2, subject_idx=subj,
        cond1="mi", cond2="non-mi", num_classes=2
    )
    all_subject_avg_accs.append(avg)

# Final summary
overall_avg = np.mean(all_subject_avg_accs)
print(f"\n✅ FINAL REPORT: Average accuracy over {half} subjects: {overall_avg:.4f}")
