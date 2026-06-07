import pickle
import os
import numpy as np

config_name = "denseTrue_layernormFalse_earlystopTrue_heads4_dim6"
subject_indices = range(5)  # adjust to total number of subjects
accuracies = []
# Define parameter options
dense_options = [False]
layernorm_options = [True, False]
earlystop_options = [True, False]
num_heads_options = [6, 8, 10]
key_dim_options = [8, 10, 12]
dense_sizes = [180, 256]

for add_dense in dense_options:
    for add_layernorm in layernorm_options:
        for use_earlystop in earlystop_options:
            for num_heads in num_heads_options:
                for key_dim in key_dim_options:
                    for Dsize in dense_sizes:
                        for subj in subject_indices:
                            config_name = f"heads{num_heads}_dim{key_dim}_denseSize{Dsize}"
                            results_file = f"5loso_subject{subj}_{config_name}.pkl"
                            filename = f"5loso_subject{subj}_{config_name}.pkl"
                            if os.path.exists(filename):
                                with open(filename, 'rb') as f:
                                    saved = pickle.load(f)
                                    acc = saved['accuracy']
                                    accuracies.append(acc)
                            else:
                                print(f"Subject {subj}: file not found")

                        if accuracies:
                            avg_acc = np.mean(accuracies)
                            print(f"{config_name}: {avg_acc:.4f}")
                        else:
                            print(f"\n❌ No accuracies found for config [{config_name}]")


