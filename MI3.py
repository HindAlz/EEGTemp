import mat73
import numpy as np
import os
from models import ATCNet
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import backend as K
import gc

gpus = tf.config.list_physical_devices('GPU')

mat = mat73.loadmat('/mnt/c/Users/Hzaab/Downloads/MIT_img_bl2anddata.mat')
proc_data = mat["data"]

print("cuda built:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

with open('processed_data_nt_st.pkl', 'rb') as f:
    X_data_nt_st, Y_data_nt_st = pickle.load(f)

with open('processed_data_nt_ct.pkl', 'rb') as f:
    X_data_nt_ct, Y_data_nt_ct = pickle.load(f)

with open('processed_data_ct_st.pkl', 'rb') as f:
    X_data_ct_st, Y_data_ct_st = pickle.load(f)

with open('processed_data_four_nonRes.pkl', 'rb') as f:
    X_data_four, Y_data_four = pickle.load(f)

with open('processed_data_three.pkl', 'rb') as f:
    X_data_three, Y_data_three = pickle.load(f)

with open("processed_data_mi_nonMi.pkl", 'rb') as f:
    X_data_list, Y_data_list = pickle.load(f)



from tensorflow.keras.utils import to_categorical
import os
import pickle
def clear_tf_memory():
    """Clears TensorFlow's GPU memory and runs garbage collection."""
    K.clear_session()
    gc.collect()
    # print("Cleared TF session and ran garbage collection.") # Optional verbose

def losoModel(subjects_X_data, subject_y_data, num_subjects, cond1, cond2):
    # File to save intermediate results
    results_file = f"loso_results_{cond1}_{cond2}.pkl"

    # Load previous progress if exists
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            saved = pickle.load(f)
        start_subject = len(saved['accuracies'])
        accuracies = saved['accuracies']
        predictions = saved['predictions']
        actual = saved['actual']
        print(f"Resuming from subject {start_subject} for {cond1}, {cond2}")
    else:
        start_subject = 0
        accuracies = []
        predictions = []
        actual = []

    for subject in range(start_subject, num_subjects):
        print("test subject:", subject)
        tf.keras.backend.clear_session()

        X_test = subjects_X_data[subject]
        y_test = subject_y_data[subject]

        X_train = np.concatenate(subjects_X_data[:subject] + subjects_X_data[subject+1:], axis=0)
        y_train = np.concatenate(subject_y_data[:subject] + subject_y_data[subject+1:], axis=0)

        y_train_cat = to_categorical(y_train, num_classes=2)
        y_test_cat = to_categorical(y_test, num_classes=2)
        clear_tf_memory()

        model = ATCNet(2, in_chans=60, in_samples=400, n_windows=3)#, eegn_dropout=0.5, tcn_dropout=0.5, eegn_F1=8, tcn_filters=16)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

        hist = model.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=1, callbacks=[callback])
        loss, acc = model.evaluate(X_test, y_test_cat, batch_size=16, verbose=1)
        print("subject ", subject, " accuracy: ", acc, " loss: ", loss)

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
    conf_normalized = conf / conf.sum(axis=1, keepdims=True)

    sns.heatmap(conf_normalized * 100, annot=True, fmt='.2f', cmap='Blues',  xticklabels=[cond2, cond1], yticklabels=[cond2, cond1], vmin=0, vmax=100)
    plt.ylabel("actual")
    plt.xlabel("predicted")
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[cond2, cond1])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    avg_acc = np.mean(accuracies)
    print("avg accuracy:", avg_acc)
    return accuracies, avg_acc


def losoModel4(subjects_X_data, subject_y_data, num_subjects):
    import os
    import pickle
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    results_file = f"loso_results_four_NoRes.pkl"

    # Load previous progress if exists
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            saved = pickle.load(f)
        start_subject = len(saved['accuracies'])
        accuracies = saved['accuracies']
        predictions = saved['predictions']
        actual = saved['actual']
        print(f"Resuming from subject {start_subject}")
    else:
        start_subject = 0
        accuracies = []
        predictions = []
        actual = []

    for subject in range(start_subject, num_subjects):
        print("test subject:", subject)
        tf.keras.backend.clear_session()

        X_test = subjects_X_data[subject]
        y_test = subject_y_data[subject]

        X_train = np.concatenate(subjects_X_data[:subject] + subjects_X_data[subject + 1:], axis=0)
        y_train = np.concatenate(subject_y_data[:subject] + subject_y_data[subject + 1:], axis=0)

        print("X_train shape:" ,X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        print("Train label distribution:", np.bincount(y_train))
        print("Test label distribution:", np.bincount(y_test))

        y_train_cat = to_categorical(y_train, num_classes=4)
        y_test_cat = to_categorical(y_test, num_classes=4)

        clear_tf_memory()

        model = ATCNet(4, in_chans=60, in_samples=400, n_windows=3)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

        hist = model.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=1, callbacks=[callback])

        loss, acc = model.evaluate(X_test, y_test_cat, batch_size=16, verbose=1)
        print("subject", subject, "accuracy:", acc, "loss:", loss)

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
    conf_normalized = conf / conf.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_normalized * 100, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=100, xticklabels=["ct", "nt", "st", "rest"], yticklabels=["ct", "nt", "st", "rest"])
    plt.show()

    # Raw confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=["ct", "nt", "st", "rest"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    avg_acc = np.mean(accuracies)
    print("Average Accuracy:", avg_acc)
    return accuracies, avg_acc

def losoModel3(subjects_X_data, subject_y_data, num_subjects):
    import os
    import pickle
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    results_file = f"loso_results_three.pkl"

    # Load previous progress if exists
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            saved = pickle.load(f)
        start_subject = len(saved['accuracies'])
        accuracies = saved['accuracies']
        predictions = saved['predictions']
        actual = saved['actual']
        print(f"Resuming from subject {start_subject}")
    else:
        start_subject = 0
        accuracies = []
        predictions = []
        actual = []

    for subject in range(start_subject, num_subjects):
        print("test subject:", subject)
        tf.keras.backend.clear_session()

        X_test = subjects_X_data[subject]
        y_test = subject_y_data[subject]

        X_train = np.concatenate(subjects_X_data[:subject] + subjects_X_data[subject + 1:], axis=0)
        y_train = np.concatenate(subject_y_data[:subject] + subject_y_data[subject + 1:], axis=0)

        print("X_train shape:" ,X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        print("Train label distribution:", np.bincount(y_train))
        print("Test label distribution:", np.bincount(y_test))

        y_train_cat = to_categorical(y_train, num_classes=3)
        y_test_cat = to_categorical(y_test, num_classes=3)

        clear_tf_memory()

        model = ATCNet(3, in_chans=60, in_samples=600, n_windows=3)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

        hist = model.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=1, callbacks=[callback])

        loss, acc = model.evaluate(X_test, y_test_cat, batch_size=16, verbose=1)
        print("subject", subject, "accuracy:", acc, "loss:", loss)

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
    conf_normalized = conf / conf.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_normalized * 100, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=100, xticklabels=["ct", "nt", "st"], yticklabels=["ct", "nt", "st"])
    plt.show()

    # Raw confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=["ct", "nt", "st"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    avg_acc = np.mean(accuracies)
    print("Average Accuracy:", avg_acc)
    return accuracies, avg_acc


num_subjects = len(X_data_nt_st)  #all conditions have same number of subjects
print(num_subjects)

#accuracies_three, avg_three = losoModel3(X_data_three, Y_data_three, len(X_data_three))
#accuracies_four, avg_four = losoModel4(X_data_four, Y_data_four, len(X_data_four))



#accuracies_nt_st, avg_acc_nt_st = losoModel(X_data_nt_st, Y_data_nt_st, len(X_data_nt_st), "nt","st")
#accuracies_nt_ct, avg_acc_nt_ct = losoModel(X_data_nt_ct, Y_data_nt_ct, len(X_data_nt_ct), "nt","ct")
#accuracies_ct_st, avg_acc_ct_st = losoModel(X_data_ct_st, Y_data_ct_st, len(X_data_ct_st), "ct","st")
accuracies_mi, avg_acc_mi = losoModel(X_data_list, Y_data_list, len(X_data_list), "mi","non-mi")


plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies_four) + 1), np.array(accuracies_four) * 100, marker='o', linestyle='-', color='blue')
plt.xlabel("subject no.")
plt.ylabel("accuracy")
plt.title("four")
plt.grid(True)
plt.xticks(range(1, len(accuracies_four) + 1))
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

'''
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies_nt_st) + 1), np.array(accuracies_nt_st) * 100, marker='o', linestyle='-', color='blue')
plt.xlabel("subject no.")
plt.ylabel("accuracy")
plt.title("nt vs st")
plt.grid(True)
plt.xticks(range(1, len(accuracies_nt_st) + 1))
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies_nt_ct) + 1), np.array(accuracies_nt_ct) * 100, marker='o', linestyle='-', color='blue')
plt.xlabel("subject no.")
plt.ylabel("accuracy")
plt.title("nt vs ct")
plt.grid(True)
plt.xticks(range(1, len(accuracies_nt_ct) + 1))
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies_ct_st) + 1), np.array(accuracies_ct_st) * 100, marker='o', linestyle='-', color='blue')
plt.xlabel("subject no.")
plt.ylabel("accuracy")
plt.title("st vs ct")
plt.grid(True)
plt.xticks(range(1, len(accuracies_ct_st) + 1))
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))

# Plot each accuracy list
plt.plot(range(1, len(accuracies_four) + 1), np.array(accuracies_four) * 100, linestyle='-', label='Four-class')
plt.plot(range(1, len(accuracies_nt_st) + 1), np.array(accuracies_nt_st) * 100,  linestyle='-', label='NT vs ST')
plt.plot(range(1, len(accuracies_nt_ct) + 1), np.array(accuracies_nt_ct) * 100, linestyle='-', label='NT vs CT')
plt.plot(range(1, len(accuracies_ct_st) + 1), np.array(accuracies_ct_st) * 100,  linestyle='-', label='ST vs CT')

# Labels and formatting
plt.xlabel("Subject No.")
plt.ylabel("Accuracy (%)")
plt.title("LOSO Accuracy per Subject for All Conditions")
plt.xticks(range(1, len(accuracies_four) + 1))
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("nt vs st accuracy = ", avg_acc_nt_st)
print("nt vs ct accuracy = ", avg_acc_nt_ct)
print("ct vs st accuracy = ", avg_acc_ct_st)

print("three: ", avg_three)
print("all four = ", avg_four)

'''
print("mi vs non-mi accuracy = ", avg_acc_mi)
