'''
Incremental LOSO with Persistent Timestamp

Description: Classification training for a model with batch normalization,
             modified for incremental LOSO processing, result appending,
             and persistent timestamp handling to ensure resumed runs
             append to the original run's files.
Should be run as 'python -m classification.classifier_bn' (if from same file hierarchy as repo)

Joseph Hong

'''

# ==================================================================================================
# IMPORTS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from datetime import datetime
import gc
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from collections import defaultdict
import random
import psutil
from sklearn.preprocessing import StandardScaler
import json # Needed for progress tracker

# Assuming models are in a 'models' subdirectory relative to the script
from models.EEGModels import EEGNet
from models.atcnet_new import ATCNet_

# from models.models import ATCNet_

# ==================================================================================================
# VARIABLES

# Seeds
SEED = 9070
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Set Model Choice ('EEGNet' or 'ATCNet')
model_choice = 'ATCNet'  # Change to 'EEGNet' if needed

# SHAP Analysis Toggle
shap_on = True  # Set to False if you don't need SHAP analysis

# FIX TENSORFLOW MEMORY GROWTH (PARTIAL FIX)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU Memory Growth Error:", e)

# Directories & Files
base_dir = "codes/"  # Get script's directory
data_dir = os.path.join(base_dir, 'data')
ref_weights_dir = os.path.join(base_dir, "reference_weights")
saved_weights_dir = os.path.join(base_dir, "saved_weights")
results_dir = os.path.join(base_dir, "results")
shap_dir = os.path.join(base_dir, "shap")
progress_file = os.path.join(base_dir, "progress_tracker.json") # Use JSON format

# --- Create Base Directories if they don't exist ---
# Specific subdirs like shap_subject_dir created later if needed
os.makedirs(saved_weights_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(shap_dir, exist_ok=True)

# ==================================================================================================
# DEFINE FILENAMES (USING THE DETERMINED TIMESTAMP)

# --- Data Configurations ---
data_type = 'bci'
data_version = 'me_v1'
if(data_type == 'mit'):
    data_filename = f"mit_subject_data_{data_version}.npz"
    weight_name = "ST"
elif (data_type == 'xon'):
    data_filename = f"xon_subject_data_{data_version}.npz"
    weight_name = "XON"
elif (data_type == 'bci'):
    data_filename = f"bci_subject_data_{data_version}.npz"
    weight_name = "BCI"
else:
    data_filename = f"subject_data_{data_version}.npz"
    weight_name = "NT"

# ==================================================================================================
# LOAD/MANAGE RUN STATE (Timestamp & Progress)

timestamp = None
start_subject_index = 0
last_processed_index = -1 # Default if starting fresh

try:
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
        # Check for essential keys before using them
        if 'timestamp' in progress_data and 'last_processed_index' in progress_data:
            timestamp = progress_data['timestamp']
            last_processed_index = int(progress_data['last_processed_index']) # Ensure it's an int
            start_subject_index = last_processed_index + 1
            print(f"--- Resuming Run ---")
            print(f"Loaded Timestamp: {timestamp}")
            print(f"Last Processed Subject Index: {last_processed_index}")
            print(f"Starting from Subject Index: {start_subject_index}")
        else:
             # File exists but keys are missing - treat as corrupted, start fresh
             print(f"Warning: Progress file {progress_file} is missing expected keys. Starting a new run.")
             raise FileNotFoundError # Treat as if file wasn't useful

except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"--- Starting New Run ---")
    print(f"Progress file {progress_file} not found or invalid ({type(e).__name__}).")
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    start_subject_index = 0
    last_processed_index = -1
    print(f"Generated New Timestamp: {timestamp}")
    print(f"Starting from Subject Index: 0")
    # Immediately save the state for this new run
    progress_data = {"timestamp": timestamp, "last_processed_index": last_processed_index}
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)
        print(f"Initialized progress file: {progress_file}")
    except Exception as write_e:
        print(f"CRITICAL ERROR: Could not write initial progress file {progress_file}: {write_e}")
        print("Exiting to prevent inconsistent state.")
        exit() # Exit if we can't even save the initial state
except Exception as e:
     print(f"CRITICAL ERROR: An unexpected error occurred reading progress file {progress_file}: {e}")
     print("Exiting to prevent inconsistent state.")
     exit()

# --- Create Timestamped Subdirectories if needed (and if shap is on) ---
if shap_on:
    # Use the loaded/generated timestamp for the directory name
    shap_subject_dir = os.path.join(shap_dir, f"{timestamp}_subject_shap_values")
    os.makedirs(shap_subject_dir, exist_ok=True)

# --- Define output file paths using the timestamp ---
accuracy_log_file = os.path.join(results_dir, f"{timestamp}_accuracy_log_LOSO.csv")
predictions_file = os.path.join(results_dir, f"{timestamp}_predictions_LOSO.npz")
misclassification_file = os.path.join(results_dir, f"{timestamp}_misclassifications_LOSO.json")


# ==================================================================================================
# LOAD DATA & PREPARE

# Load Data
try:
    data_path = os.path.join(data_dir, data_filename)
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    subject_ids = data['subject_ids']
    print(f"\nData loaded from {data_path}.")
    print(f"X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Data file not found at {data_path}")
    exit()
except Exception as e:
    print(f"CRITICAL ERROR: Error loading data: {e}")
    exit()


# Standard scaling per subject (Keep this as is based on original script)
scalers = {}
unique_subject_list = np.unique(subject_ids)
print(f"Found {len(unique_subject_list)} unique subjects.")
for subj in unique_subject_list:
    subj_idx = np.where(subject_ids == subj)[0]
    if len(subj_idx) == 0:
        print(f"Warning: No data found for subject {subj}. Skipping scaling.")
        continue
    original_shape = X[subj_idx].shape
    X_subj = X[subj_idx].reshape(original_shape[0], -1)
    scaler = StandardScaler()
    try:
        X_subj_scaled = scaler.fit_transform(X_subj)
        X[subj_idx] = X_subj_scaled.reshape(original_shape)
        scalers[subj] = scaler
    except ValueError as e:
        print(f"Warning: Error scaling data for subject {subj}: {e}. Using unscaled data for this subject.")

# LOSO Parameters
n_splits = len(unique_subject_list)
epochs = 70
batch_size = 16
learning_rate = 0.00005
nb_classes = len(np.unique(y))
weight_decay = 0.01
print(f"Number of classes: {nb_classes}")

# ==================================================================================================
# INITIALIZE/LOAD RESULTS FILES

# --- Initialize Accuracy Log File (write header if new or run is new) ---
if start_subject_index == 0 or not os.path.exists(accuracy_log_file): # Write header only if starting fresh
    try:
        with open(accuracy_log_file, 'w') as f:
            f.write("Subject,Loss,Accuracy\n") # CSV Header
        print(f"Initialized accuracy log file: {accuracy_log_file}")
    except Exception as e:
         print(f"Warning: Could not initialize accuracy log file {accuracy_log_file}: {e}")


# --- Load existing predictions if available ---
all_predictions = {}
if os.path.exists(predictions_file):
     # Only load if we are resuming a run (start_subject_index > 0)
     if start_subject_index > 0:
         try:
             loaded_data = np.load(predictions_file, allow_pickle=True)
             if 'predictions' in loaded_data:
                 all_predictions = loaded_data['predictions'].item()
                 print(f"Loaded existing predictions from: {predictions_file}")
             else:
                 print(f"Warning: Predictions file {predictions_file} has unexpected format. Starting predictions dictionary fresh.")
         except Exception as e:
              print(f"Warning: Could not load existing predictions file {predictions_file}. Starting predictions dictionary fresh. Error: {e}")
     else:
          # If starting a new run (index 0), ignore existing file content for this dictionary
          print(f"Starting new run, initializing predictions dictionary fresh (ignoring any content in {predictions_file}).")
else:
     # If file doesn't exist, always start fresh
     print(f"Predictions file {predictions_file} not found. Initializing predictions dictionary fresh.")


# --- Load existing misclassifications if available ---
all_misclassifications = {}
if os.path.exists(misclassification_file):
    if start_subject_index > 0: # Only load if resuming
        try:
            with open(misclassification_file, 'r') as f:
                all_misclassifications = json.load(f)
            print(f"Loaded existing misclassifications from: {misclassification_file}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode existing misclassification file {misclassification_file}. Starting misclassifications dictionary fresh.")
            all_misclassifications = {}
        except Exception as e:
            print(f"Warning: Could not load existing misclassification file {misclassification_file}. Starting misclassifications dictionary fresh. Error: {e}")
            all_misclassifications = {}
    else:
        print(f"Starting new run, initializing misclassifications dictionary fresh (ignoring any content in {misclassification_file}).")
else:
     print(f"Misclassification file {misclassification_file} not found. Initializing misclassifications dictionary fresh.")


# Evaluation Tracking (Only confusion matrix needs accumulation in memory)
conf_matrix_accum = np.zeros((nb_classes, nb_classes))
# Try to load accumulated matrix if resuming, though recalculating might be safer
# If needed, could save/load conf_matrix_accum to/from a file too. For now, it recalculates on resume.


# ==================================================================================================
# UTILITY FUNCTIONS
def print_memory_usage():
    """Prints current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    print(f"🛑 Memory Usage: {mem_usage:.2f} GB")

def clear_tf_memory():
    """Clears TensorFlow's GPU memory and runs garbage collection."""
    K.clear_session()
    gc.collect()
    # print("Cleared TF session and ran garbage collection.") # Optional verbose


# ==================================================================================================
# TRAINING LOOP (Incremental LOSO with Persistent Timestamp)

print(f"\n===== Starting LOSO Loop from Subject Index {start_subject_index} =====")
if start_subject_index >= len(unique_subject_list):
     print("Start index is beyond the number of subjects. Nothing to process.")

for i in range(start_subject_index, len(unique_subject_list)):
    subject = unique_subject_list[i]
    print(f"\n===== Processing Subject {subject} (Index: {i}) =====")
    print_memory_usage()

    # --- Prepare Data for Current Fold ---
    test_index = np.where(subject_ids == subject)[0]
    train_index = np.where(subject_ids != subject)[0]

    X_train_subj = X[train_index]
    y_train_subj = y[train_index]
    X_test_subj = X[test_index]
    y_test_subj = y[test_index]

    # Reshape input for models
    if model_choice == 'ATCNet':
        # ATCNet expects (trials, 1, channels, timepoints)
        X_train = np.expand_dims(X_train_subj, axis=1)
        X_test = np.expand_dims(X_test_subj, axis=1)
    elif model_choice == 'EEGNet':
         # EEGNet standard: (trials, channels, timepoints, 1)
         X_train = np.expand_dims(X_train_subj, axis=-1)
         X_test = np.expand_dims(X_test_subj, axis=-1)
    else:
         X_train = X_train_subj # Default
         X_test = X_test_subj

    y_train = to_categorical(y_train_subj, nb_classes)
    y_test = to_categorical(y_test_subj, nb_classes)


    # --- Initialize and Compile Model ---
    print(f"Input shape for model: X_train: {X_train.shape}, X_test: {X_test.shape}")
    clear_tf_memory() # Clear previous model graph from memory

    if model_choice == 'ATCNet':
        # Ensure ATCNet params match input shape: (trials, 1, chans, samples)
        model = ATCNet_(nb_classes, X_train.shape[2], X_train.shape[3])
    elif model_choice == 'EEGNet':
         # Ensure EEGNet params match input shape: (trials, chans, samples, 1)
         model = EEGNet(nb_classes, X_train.shape[1], X_train.shape[2])
    else:
         print(f"Error: Unknown model_choice '{model_choice}'")
         continue # Skip subject

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --- Callbacks ---
    # Use the loaded/generated timestamp in the checkpoint filename
    model_checkpoint_path = os.path.join(saved_weights_dir, f"{timestamp}_best_model_subject_{subject}.weights.h5")
    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=False, save_weights_only=True, verbose=0)
    ]

    print_memory_usage()

    # --- Train Model ---
    print(f"Training model for subject {subject}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    print("Training finished.")
    print_memory_usage()

    # --- Load Best Weights ---
    try:
         print(f"Loading best weights from {model_checkpoint_path}")
         model.load_weights(model_checkpoint_path)
    except Exception as e:
         print(f"Warning: Could not load best weights for subject {subject} from {model_checkpoint_path}. Using weights from end of training. Error: {e}")

    # --- Evaluate Model ---
    print(f"Evaluating model for subject {subject}...")
    scores = model.evaluate(X_test, y_test, verbose=0)
    subject_loss = scores[0]
    subject_accuracy = scores[1]
    print(f"Subject {subject} - Loss: {subject_loss:.4f}, Accuracy: {subject_accuracy * 100:.2f}%")

    # --- Append Accuracy Results ---
    try:
        # Open in append mode ('a')
        with open(accuracy_log_file, 'a') as f:
            f.write(f"{subject},{subject_loss:.6f},{subject_accuracy:.6f}\n")
    except Exception as e:
        print(f"Error writing to accuracy log file: {e}")

    # --- Predict & Get Subject CM ---
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    conf_matrix_subject = tf.math.confusion_matrix(
        y_test_classes, y_pred_classes, num_classes=nb_classes
    ).numpy()

    # --- Append Predictions and Subject CM ---
    subject_predictions = {
        'y_test': y_test_classes.astype(np.int32),
        'y_pred_prob': y_pred_prob.astype(np.float32),
        'confusion_matrix': conf_matrix_subject.astype(np.int32)
    }
    all_predictions[subject] = subject_predictions # Add/update subject's data
    try:
         # Save the entire updated dictionary to the NPZ file
         np.savez_compressed(predictions_file, predictions=all_predictions)
         print(f"Saved predictions & CM for subject {subject} to {predictions_file}") # Optional verbose
    except Exception as e:
         print(f"Error saving predictions/CM to NPZ file: {e}")

    # --- SHAP Analysis (if enabled) ---
    if shap_on:

        print("Computing SHAP values...")
        try:
            # Patches...
            shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
            shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.
            shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.
            shap.explainers._deep.deep_tf.op_handlers["DepthwiseConv2dNative"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.
            shap.explainers._deep.deep_tf.op_handlers["BatchToSpaceND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.
            shap.explainers._deep.deep_tf.op_handlers["SpaceToBatchND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.
            shap.explainers._deep.deep_tf.op_handlers["Einsum"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.
            shap.explainers._deep.deep_tf.op_handlers["BatchMatMulV2"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.
            shap.explainers._deep.deep_tf.op_handlers["Neg"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.

            # num_background_samples = min(300, X_train.shape[0])
            # background_indices = np.random.choice(X_train.shape[0], num_background_samples, replace=False)
            # background = X_train[background_indices]

            # Set up 300 random points for shap
            background = np.array(X_train[np.random.choice(X_train.shape[0], 300, replace=False)])
            print(np.shape(X_test))

            e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(X_test, check_additivity=False)

            # Use the loaded/generated timestamp in the SHAP filename
            # shap_subject_dir should already exist if shap_on is True
            shap_filename = os.path.join(shap_subject_dir, f"subject_{subject}_shap.pkl") # Timestamp is in parent dir name
            try:
                with open(shap_filename, "wb") as fp:
                    pickle.dump(shap_values, fp)
                # print(f"Saved SHAP values for subject {subject} to {shap_filename}")
            except Exception as e:
                print(f"Error saving SHAP values for subject {subject}: {e}")
            del e, shap_values, background
            gc.collect()
        except Exception as e:
            print(f"ERROR during SHAP analysis for subject {subject}: {e}")

    # --- Misclassification Tracking ---
    misclassified_indices = np.where(y_pred_classes != y_test_classes)[0]
    subject_key = str(subject) # Use string key for JSON
    all_misclassifications[subject_key] = misclassified_indices.tolist() # Add/update subject's data
    try:
        # Save the entire updated dictionary
        with open(misclassification_file, 'w') as f:
            json.dump(all_misclassifications, f, indent=4)
    except Exception as e:
        print(f"Error writing misclassification data: {e}")

    # --- Update Accumulated Confusion Matrix ---
    conf_matrix_accum += conf_matrix_subject

    # --- Clean Up Memory ---
    del model, X_train, y_train, X_test, y_test, X_train_subj, y_train_subj, X_test_subj, y_test_subj
    del y_pred_prob, y_pred_classes, y_test_classes, conf_matrix_subject, subject_predictions
    del history, callbacks # Explicitly delete large objects
    # print_memory_usage()
    clear_tf_memory()
    gc.collect()
    print(f"===== Finished processing Subject {subject} =====")

    # --- Update Progress Tracker ---
    # Save the CURRENT timestamp and the index of the subject JUST COMPLETED
    last_processed_index = i
    progress_data = {"timestamp": timestamp, "last_processed_index": last_processed_index}
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)
    except Exception as e:
        print(f"Error updating progress tracker file {progress_file}: {e}")


# ==================================================================================================
# POST-PROCESSING (After loop finishes naturally or is interrupted and restarted later)

print("\n===== LOSO Loop Processing Ended =====")
num_subjects_processed = last_processed_index + 1 # How many were actually done

if num_subjects_processed > 0:
    # --- Calculate and Print Average Accuracy from Log File ---
    try:
        results_df = pd.read_csv(accuracy_log_file)
        # Filter results for the current timestamp if multiple runs are logged (though they shouldn't be with this logic)
        # No, the file itself is timestamped, so all entries belong to this run.
        if not results_df.empty:
            # Ensure we only calculate average based on subjects actually processed in this run
            processed_subjects_in_log = results_df['Subject'].nunique()
            if processed_subjects_in_log != num_subjects_processed:
                 print(f"Warning: Number of subjects in log ({processed_subjects_in_log}) doesn't match expected ({num_subjects_processed}). Average might be inaccurate if file was manually edited.")

            average_accuracy = results_df['Accuracy'].mean()
            average_loss = results_df['Loss'].mean()
            print(f"Average Accuracy across {processed_subjects_in_log} subjects in log: {average_accuracy * 100:.2f}%")

            # --- Save Summary Stats ---
            summary_file = os.path.join(results_dir, f"{timestamp}_summary_LOSO.txt")
            with open(summary_file, "w") as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Model: {model_choice}\n")
                f.write(f"Data Type: {data_type} ({data_filename})\n")
                f.write(f"Number of Subjects Processed in this run: {num_subjects_processed}\n")
                f.write(f"Number of Subjects in Accuracy Log: {processed_subjects_in_log}\n")
                f.write(f"Average Loss (from log): {average_loss:.4f}\n")
                f.write(f"Average Accuracy (from log): {average_accuracy * 100:.2f}%\n")
                f.write("\nAccumulated Confusion Matrix (based on processed subjects):\n")
                f.write(np.array2string(conf_matrix_accum.astype(int), precision=0))
                f.write("\n\n--- Output Files ---")
                f.write("\nAccuracy Log File: " + accuracy_log_file)
                f.write("\nPredictions & Individual CMs File: " + predictions_file)
                f.write("\nMisclassifications File: " + misclassification_file)
                if shap_on:
                    f.write("\nSHAP Values Directory: " + shap_subject_dir) # Use correct var
                f.write("\n\nNote: Individual subject confusion matrices are stored within the predictions NPZ file.")
            print(f"Summary saved to {summary_file}")
        else:
            print(f"Accuracy log file {accuracy_log_file} is empty. Cannot calculate average.")
    except FileNotFoundError:
        print(f"Accuracy log file not found at {accuracy_log_file}. Cannot calculate average.")
    except Exception as e:
        print(f"Error reading or processing accuracy log file: {e}")

    # --- Save Accumulated Confusion Matrix Plot ---
    if np.sum(conf_matrix_accum) > 0:
        try:
            plt.figure(figsize=(8, 6))
            # Check for division by zero before normalizing
            row_sums = conf_matrix_accum.sum(axis=1)[:, np.newaxis]
            # Handle rows with zero sum (e.g., if a class never appeared in test sets)
            # Replace 0s in denominator with 1s to avoid division by zero, result will be 0 anyway
            safe_row_sums = np.where(row_sums == 0, 1, row_sums)
            conf_matrix_norm = conf_matrix_accum.astype('float') / safe_row_sums
            # conf_matrix_norm = np.nan_to_num(conf_matrix_norm) # Alternative if not using safe_row_sums

            sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(nb_classes), yticklabels=range(nb_classes))
            plt.title(f"Accumulated Normalized Confusion Matrix ({timestamp})")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            conf_matrix_filename = os.path.join(results_dir, f"{timestamp}_confusion_matrix.png")
            plt.savefig(conf_matrix_filename)
            plt.close()
            print(f"Accumulated confusion matrix plot saved to {conf_matrix_filename}")
        except Exception as e:
            print(f"Error saving accumulated confusion matrix plot: {e}")
    else:
        print("Accumulated confusion matrix is empty (no subjects processed?). Skipping plot.")
else:
     print("No subjects were processed in this run. Skipping summary generation and confusion matrix plot.")


# --- Check for Full Completion ---
fully_completed = (last_processed_index == len(unique_subject_list) - 1)
if fully_completed:
     print("\n***** LOSO process fully completed for all subjects. *****")
     try:
          if os.path.exists(progress_file):
               os.remove(progress_file)
               print(f"Removed progress tracker file: {progress_file}")
     except Exception as e:
          print(f"Error removing progress tracker file: {e}")


# ==================================================================================================
# FINAL TRAINING ON ALL DATA (Only if LOSO fully completed)

if fully_completed:
    print("\n===== Training Final Model on All Subjects... =====")
    clear_tf_memory()
    print_memory_usage()

    if model_choice == 'ATCNet':
        final_model = ATCNet_(nb_classes, X.shape[1], X.shape[2])
    elif model_choice == 'EEGNet':
        final_model = EEGNet(nb_classes, X.shape[1], X.shape[2])
    else:
        print(f"Error: Unknown model_choice '{model_choice}' for final model.")
        final_model = None

    if final_model:
        # Prepare all data
        if model_choice == 'ATCNet':
            X_all = np.expand_dims(X, axis=1)
        elif model_choice == 'EEGNet':
            X_all = np.expand_dims(X, axis=-1)
        else:
             X_all = X
        y_all = to_categorical(y, nb_classes)
        print(f"Final training data shape: X_all: {X_all.shape}, y_all: {y_all.shape}")

        # Compile Final Model
        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Train Final Model
        print("Fitting final model...")
        final_model.fit(X_all, y_all, batch_size=batch_size, epochs=epochs, verbose=1)

        # Save Final Model Weights using the run's timestamp
        final_weights_path = os.path.join(saved_weights_dir, f"{timestamp}_FINAL_{model_choice}_{weight_name}.weights.h5")
        try:
            final_model.save_weights(final_weights_path)
            print(f"Final Model Weights Saved to {final_weights_path}")
        except Exception as e:
            print(f"Error saving final model weights: {e}")

        del final_model, X_all, y_all
        clear_tf_memory()
        gc.collect()
else:
     print("\nLOSO process did not complete fully. Skipping final model training.")


print("\n===== Script Finished =====")