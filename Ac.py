# --- Imports ---
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
    BatchNormalization, Input, Reshape, MultiHeadAttention, GlobalAveragePooling1D, DepthwiseConv2D
)
from tensorflow.keras.constraints import max_norm
from scipy.signal import welch

def clear_tf_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def temporal(in_chans, in_samples, num_classes):

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

def Psd(data):
    f, Pxx_den = welch(data[2][1], fs=1, nperseg=1024)
    plt.semilogy(f, Pxx_den)
    plt.ylim([0.5e-3, 1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V²/Hz]')
    plt.show()



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
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

def losoModelT(subjects_X_data, subject_y_data, num_subjects, cond1, cond2, num_classes):
    results_file = f"loso_results_three_Temporal_{cond1}_{cond2}.pkl"

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

        model = Spatial(in_chans=60, in_samples=600, num_classes=num_classes)
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
                xticklabels=[cond2, cond1], yticklabels=[cond2, cond1], vmin=0, vmax=100)
    plt.title("Normalized Confusion Matrix")
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

def losoModelS(subjects_X_data, subject_y_data, num_subjects, cond1, cond2, num_classes):
    results_file = f"loso_results_three_Spatial_{cond1}_{cond2}.pkl"

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

        model = Spatial(in_chans=60, in_samples=600, num_classes=num_classes)
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
                xticklabels=[cond2, cond1], yticklabels=[cond2, cond1], vmin=0, vmax=100)
    plt.title("Normalized Confusion Matrix")
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



with open("processed_data_three.pkl", 'rb') as f:
    X_three_data_list, Y_three_data_list = pickle.load(f)

num_subjects = len(X_three_data_list)
print(num_subjects)

losoModel(X_three_data_list, Y_three_data_list, num_subjects, cond1="mi", cond2="nonMi", num_classes=3)
losoModelS(X_three_data_list, Y_three_data_list, num_subjects, cond1="mi", cond2="nonMi", num_classes=3)

#compare 3 classes, atc net with 2 class system