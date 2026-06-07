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
    BatchNormalization, Input, Reshape, MultiHeadAttention, GlobalAveragePooling1D, GlobalAveragePooling2D, DepthwiseConv2D
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


def Temporal(in_chans, in_samples, num_classes):
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

    info = x.shape[2] * x.shape[3]
    x = Reshape((x.shape[1], info))(x)  #(none, 75, 480)

    x = MultiHeadAttention(num_heads=2, key_dim=3)(x, x)
    x = GlobalAveragePooling1D()(x)

    return Model(inputs=inputs, outputs=x)


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

    info = x.shape[2] * x.shape[3]
    x = Reshape((x.shape[1], info))(x)  #(none, 75, 480)

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

    info = x.shape[2] * x.shape[3]
    x = Reshape((x.shape[1], info))(x)

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
from tensorflow.keras.layers import Lambda


def threeBranch(input_shape, input_shape_psd, num_classes):
    inp = Input(shape=input_shape)  #(1, chans, time)
    in_psd = Input(shape=input_shape_psd)  #(1, chans, freqs)

    Q = Temporal(in_chans=input_shape[1], in_samples=input_shape[2], num_classes=num_classes)(inp)
    K = Spatial(in_chans=input_shape[1], in_samples=input_shape[2], num_classes=num_classes)(inp)
    V = Spectral(in_chans=input_shape_psd[1], in_samples=input_shape_psd[2], num_classes=num_classes)(in_psd)

    #(none, 1, features), normal expand cant be used, input is a keras tensor, function -> tensorflow
    Q = Lambda(lambda x: tf.expand_dims(x, axis=1))(Q)
    K = Lambda(lambda x: tf.expand_dims(x, axis=1))(K)
    V = Lambda(lambda x: tf.expand_dims(x, axis=1))(V)

    attn = MultiHeadAttention(num_heads=2, key_dim=3)(Q, K, V) #test other vals
    x = GlobalAveragePooling1D()(attn)

    x = Dense(64, activation='relu')(x) #try elu?
    x = Dropout(0.3)(x) #try other vals
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=[inp, in_psd], outputs=outputs)

def losoModel(subjects_X_data, subject_y_data, num_subjects, cond1, cond2, num_classes, sfreq=200):
    results_file = f"loso_results_mi_nonMi.pkl"

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            saved = pickle.load(f)
        start_subject = len(saved['accuracies'])
        accuracies = saved['accuracies']
        predictions = saved['predictions']
        actual = saved['actual']
    else:
        start_subject = 0
        accuracies, predictions, actual = [], [], []

    for subject in range(start_subject, num_subjects):
        print("Testing subject: ",subject)
        clear_tf_memory()

        X_train_list = []
        y_train_list = []
        for i in range(num_subjects):
            if i == subject: #skip test subj
                continue
            subjects_X_data, subject_y_data = averaging(subjects_X_data, subject_y_data)
            data_i = subjects_X_data[i].squeeze(1).transpose(1, 2, 0) #rearrange to fit psd
            psd_i = psd(data_i, fs=sfreq)
            X_train_list.append(np.expand_dims(psd_i, axis=1)) #bring back extra dim for cnn
            y_train_list.append(subject_y_data[i])

        X_train_psd = np.concatenate(X_train_list, axis=0)

        X_train = np.concatenate(subjects_X_data[:subject] + subjects_X_data[subject + 1:], axis=0)
        y_train = np.concatenate(subject_y_data[:subject] + subject_y_data[subject + 1:], axis=0)

        print("x train: ", X_train.shape) #(5444, 1, 60, 400)

        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        model = threeBranch(input_shape=X_train.shape[1:],
                              input_shape_psd=X_train_psd.shape[1:],
                              num_classes=num_classes)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([X_train, X_train_psd],y_train_cat, epochs=30, batch_size=16, verbose=1)
        loss, acc = model.evaluate([X_train, X_train_psd],y_train_cat, batch_size=16, verbose=1)


        print(f"Subject {subject} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        accuracies.append(acc)
        with open(results_file, 'wb') as f:
            pickle.dump({
                'accuracies': accuracies,
            }, f)

    conf = confusion_matrix(actual, predictions)
    conf_norm = conf / conf.sum(axis=1, keepdims=True)

    sns.heatmap(conf_norm * 100, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[cond2, cond1 ], yticklabels=[cond2, cond1 ], vmin=0, vmax=100)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[cond2, cond1 ])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Raw Confusion Matrix")
    plt.show()

    avg_acc = np.mean(accuracies)
    print("Average Accuracy:", avg_acc)
    return accuracies, avg_acc


def averaging(X_data, Y_data):
    output = []
    labels = []
    n_trials = len(Y_data)
    start = 0

    for cls in range(3):
        end = start
        while end < n_trials and Y_data[end] == cls:
            end += 1

        for i in range(start, end):
            cls_indices = np.arange(start, end)
            selected = np.random.choice(cls_indices, 5)
            avg_sample = np.mean(X_data[selected], axis=0)
            labels.append(cls)
            output.append(avg_sample)
        start = end

    return np.array(output), np.array(labels)


with open("processed_data_four.pkl", 'rb') as f:
    X_data_list, Y_data_list = pickle.load(f)


num_subjects = len(X_data_list)
print(Y_data_list[1]) #(trials, 1, channels, time (400))
#losoModel(X_data_list, Y_data_list, num_subjects, cond1="mi", cond2="non-mi", num_classes=2)
#Psd(X_data_list[0][0])
#Spectral