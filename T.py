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

    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


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

    x = Dense(32, activation='relu')(x)
    print(x.shape)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

def losoModelT(subjects_X_data, subject_y_data, num_subjects, cond1, cond2, num_classes):
    results_file = f"loso_results_three_Temporal_mi_nonMi.pkl"

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            saved = pickle.load(f)
        start_subject = len(saved['accuracies'])
        accuracies = saved['accuracies']
        predictions = saved['predictions']
        actual = saved['actual']
        print(f"Resuming from subject {start_subject} for {cond1} vs {cond2}")
    else:
        start_subject = 0
        accuracies, predictions, actual = [], [], []

    for subject in range(start_subject, num_subjects):
        print(f"\nTesting subject: {subject}")
        clear_tf_memory()

        X_test = subjects_X_data[subject]
        y_test = subject_y_data[subject]
        X_train = np.concatenate(subjects_X_data[:subject] + subjects_X_data[subject + 1:], axis=0)
        y_train = np.concatenate(subject_y_data[:subject] + subject_y_data[subject + 1:], axis=0)

        print("x train: ", X_train.shape) #(5444, 1, 60, 400)

        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        model = Temporal(in_chans=60, in_samples=400, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)

        model.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=1, callbacks=[callback])

        loss, acc = model.evaluate(X_test, y_test_cat, batch_size=16, verbose=1)
        print(f"Subject {subject} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        accuracies.append(acc)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        predictions.extend(y_pred)
        actual.extend(y_test)

        with open(results_file, 'wb') as f:
            pickle.dump({
                'accuracies': accuracies,
                'predictions': predictions,
                'actual': actual
            }, f)

    conf = confusion_matrix(actual, predictions)
    conf_norm = conf / conf.sum(axis=1, keepdims=True)

    sns.heatmap(conf_norm * 100, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[cond2,  cond1, "st" ], yticklabels=[cond2,  cond1, "st" ], vmin=0, vmax=100)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[cond2,  cond1, "st" ])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Raw Confusion Matrix")
    plt.show()

    avg_acc = np.mean(accuracies)
    print("Average Accuracy:", avg_acc)
    return accuracies, avg_acc


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

    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)



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


def losoModelSPecPSD(subjects_X_data, subject_y_data, num_subjects, cond1, cond2, num_classes, sfreq=200):
    results_file = f"loso_results_three_Spectral_PSD_mi_nonMi_400.pkl"

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            saved = pickle.load(f)
        start_subject = len(saved['accuracies'])
        accuracies = saved['accuracies']
        predictions = saved['predictions']
        actual = saved['actual']
        print(f"Resuming from subject {start_subject} for {cond1} vs {cond2}")
    else:
        start_subject = 0
        accuracies, predictions, actual = [], [], []

    for subject in range(start_subject, num_subjects):
        print(f"\nTesting subject: {subject}")
        clear_tf_memory()

        #(trials, 1, chans, time)
        test_data = subjects_X_data[subject].squeeze(1).transpose(1, 2, 0)  # (chans, time, trials)
        X_test = psd(test_data, fs=sfreq) #(trials, chans, freqs)
        X_test = np.expand_dims(X_test, axis=1)  #(trials, 1, channels, freqs)
        y_test = subject_y_data[subject]

        X_train_list = []
        y_train_list = []
        for i in range(num_subjects):
            if i == subject: #skip test subj
                continue
            data_i = subjects_X_data[i].squeeze(1).transpose(1, 2, 0) #rearrange to fit psd
            psd_i = psd(data_i, fs=sfreq)
            X_train_list.append(np.expand_dims(psd_i, axis=1)) #bring back extra dim for cnn
            y_train_list.append(subject_y_data[i])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        #print("X_train:", X_train.shape)  #(trials, 1, chans, freqs)

        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        model = Spectral(in_chans=X_train.shape[2], in_samples=X_train.shape[3], num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=1)

        loss, acc = model.evaluate(X_test, y_test_cat, batch_size=16, verbose=1)
        print(f"Subject {subject} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        accuracies.append(acc)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        predictions.extend(y_pred)
        actual.extend(y_test)

        with open(results_file, 'wb') as f:
            pickle.dump({
                'accuracies': accuracies,
                'predictions': predictions,
                'actual': actual
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

def feature_fft(matrix, period=1., mains_f=50.,
                filter_mains=True, filter_DC=True,
                normalise_signals=True,
                ntop=10, get_power_spectrum=True):
    """
    Computes the FFT of each signal.

    Parameters:
        matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the
        values of nsignals for a time window of length nsamples
        period (float): width (in seconds) of the time window represented by
        matrix
        mains_f (float): the frequency of mains power supply, in Hz.
        filter_mains (bool): should the mains frequency (plus/minus 1Hz) be
        filtered out?
        filter_DC (bool): should the DC component be removed?
        normalise_signals (bool): should the signals be normalised to the
        before interval [-1, 1] before computing the FFT?
        ntop (int): how many of the "top N" most energetic frequencies should
        also be returned (in terms of the value of the frequency, not the power)
        get_power_spectrum (bool): should the full power spectrum of each
        signal be returned (in terms of magnitude of each frequency component)

    Returns:
        numpy.ndarray: 1D array containing the ntop highest-power frequencies
        for each signal, plus (if get_power_spectrum is True) the magnitude of
        each frequency component, for all signals.
        list: list containing feature names for the quantities calculated. The
        names associated with the power spectrum indicate the frequencies down
        to 1 decimal place.

    Author:
        Original: [fcampelo]
    """

    # Signal properties
    N = matrix.shape[0]  # number of samples
    T = period / N  # Sampling period

    # Scale all signals to interval [-1, 1] (if requested)
    if normalise_signals:
        matrix = -1 + 2 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

    # Compute the (absolute values of the) FFT
    # Extract only the first half of each FFT vector, since all the information
    # is contained there (by construction the FFT returns a symmetric vector).
    fft_values = np.abs(scipy.fft.fft(matrix, axis=0))[0:N // 2] * 2 / N

    # Compute the corresponding frequencies of the FFT components
    freqs = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    # Remove DC component (if requested)
    if filter_DC:
        fft_values = fft_values[1:]
        freqs = freqs[1:]

    # Remove mains frequency component(s) (if requested)
    if filter_mains:
        indx = np.where(np.abs(freqs - mains_f) <= 1)
        fft_values = np.delete(fft_values, indx, axis=0)
        freqs = np.delete(freqs, indx)

    # Extract top N frequencies for each signal
    indx = np.argsort(fft_values, axis=0)[::-1]
    indx = indx[:ntop]

    ret = freqs[indx].flatten(order='F')

    # Make feature names
    names = []
    for i in np.arange(fft_values.shape[1]):
        names.extend(['topFreq_' + str(j) + "_" + str(i) for j in np.arange(1, 11)])

    if (get_power_spectrum):
        ret = np.hstack([ret, fft_values.flatten(order='F')])

        for i in np.arange(fft_values.shape[1]):
            names.extend(['freq_' + "{:03d}".format(int(j)) + "_" + str(i) for j in 10 * np.round(freqs, 1)])

    return ret, names
def Spectralfft(in_chans, in_samples, num_classes):
    inputs = tf.keras.Input(shape=(1, in_samples, 1))  # your input shape

    # permute swaps axes (channels_last expected?), check if needed.
    F1 = 8  # number of filters
    kernLength = 64
    poolSize = 8
    dropout = 0.4

    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False)(inputs)  # kernel: (64,1)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Fix pooling: pool along width dimension, not height
    x = AveragePooling2D(pool_size=(1, poolSize))(x)  # pool (height=1, width=8)
    x = Dropout(dropout)(x)

    # Shape after pooling: (None, 1, new_width, F1)
    info = x.shape[2] * x.shape[3]  # width * filters
    x = Reshape((x.shape[1], info))(x)  # (None, 1, width*filters)

    x = MultiHeadAttention(num_heads=2, key_dim=3)(x, x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
def losoModelSpecFFT(subjects_X_data, subject_y_data, num_subjects, cond1, cond2, num_classes, sfreq=200):
    results_file = f"loso_results_three_Spectral_FFT_three.pkl"

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            saved = pickle.load(f)
        start_subject = len(saved['accuracies'])
        accuracies = saved['accuracies']
        predictions = saved['predictions']
        actual = saved['actual']
        print(f"Resuming from subject {start_subject} for {cond1} vs {cond2}")
    else:
        start_subject = 0
        accuracies, predictions, actual = [], [], []

    for subject in range(start_subject, num_subjects):
        print(f"\nTesting subject: {subject}")
        clear_tf_memory()

        test_data = subjects_X_data[subject].squeeze(1)  # (trials, chans, time)
        # Extract FFT features for test set
        X_test = extract_fft_features(test_data, period=test_data.shape[2] / sfreq)
        X_test = np.expand_dims(X_test, axis=1)  # (N, 1, features)
        X_test = np.expand_dims(X_test, axis=-1)  # (N, 1, features, 1)

        y_test = subject_y_data[subject]

        # Prepare training data
        X_train_list = []
        y_train_list = []
        for i in range(num_subjects):
            if i == subject:
                continue
            data_i = subjects_X_data[i].squeeze(1)  # (trials, chans, time)
            X_train_i = extract_fft_features(data_i, period=data_i.shape[2] / sfreq)
            X_train_i = np.expand_dims(X_train_i, axis=1)  # (N, 1, features)
            X_train_i = np.expand_dims(X_train_i, axis=-1)  # (N, 1, features, 1)
            X_train_list.append(X_train_i)
            y_train_list.append(subject_y_data[i])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        print("X_train shape:", X_train.shape)  # (trials, 1, features)

        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        # Model input channels = 1 (since FFT features are concatenated), input samples = features dim
        model = Spectralfft(in_chans=X_train.shape[1], in_samples=X_train.shape[2], num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train_cat.shape)
        print("y_test shape:", y_test_cat.shape)

        print( X_train.shape)
        model.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=1)

        loss, acc = model.evaluate(X_test, y_test_cat, batch_size=16, verbose=1)
        print(f"Subject {subject} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        accuracies.append(acc)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        predictions.extend(y_pred)
        actual.extend(y_test)

        with open(results_file, 'wb') as f:
            pickle.dump({
                'accuracies': accuracies,
                'predictions': predictions,
                'actual': actual
            }, f)

    conf = confusion_matrix(actual, predictions)
    conf_norm = conf / conf.sum(axis=1, keepdims=True)

    sns.heatmap(conf_norm * 100, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[cond2, cond1], yticklabels=[cond2, cond1], vmin=0, vmax=100)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[cond2, cond1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Raw Confusion Matrix")
    plt.show()

    avg_acc = np.mean(accuracies)
    print("Average Accuracy:", avg_acc)
    return accuracies, avg_acc


import numpy as np

def extract_fft_features(eeg_data, period=1., mains_f=50., filter_mains=True, filter_DC=True,
                         normalise_signals=True, ntop=10, get_power_spectrum=True):
    all_features = []

    for trial in range(eeg_data.shape[0]):

        trial_data = eeg_data[trial].T

        feats, _ = feature_fft(trial_data, period=period, mains_f=mains_f,
                                   filter_mains=filter_mains, filter_DC=filter_DC,
                                   normalise_signals=normalise_signals, ntop=ntop,
                                   get_power_spectrum=get_power_spectrum)
        all_features.append(feats)

    features = np.stack(all_features)  # shape (trials, n_features)
    return features


with open("processed_data_mi_nonMi.pkl", 'rb') as f:
    X_data_list, Y_data_list = pickle.load(f)


num_subjects = len(X_data_list)
print(num_subjects)
losoModelSPecPSD(X_data_list, Y_data_list, num_subjects, cond1="mi", cond2="non-mi", num_classes=2)
#losoModelSpecFFT(X_data_list, Y_data_list, num_subjects, cond1="mi", cond2="non-mi", num_classes=3)
#Psd(X_data_list[0][0])
#Spectral