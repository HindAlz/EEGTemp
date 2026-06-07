import pickle
import os
import numpy as np

config_name = "denseTrue_layernormFalse_earlystopTrue_heads4_dim6"
subject_indices = range(5)  # adjust to total number of subjects
accuracies = []
# Define parameter options
dense_options = [True]
layernorm_options = [True]
earlystop_options = [True]
num_heads_options = [2,3, 4,5,6,7,8,9]
key_dim_options = [10]
dense_sizes = [256]
activations = ["elu"]
attns = [['Q', 'K', 'V']]

for add_dense in dense_options:
    for add_layernorm in layernorm_options:
        for use_earlystop in earlystop_options:
            for avgs in num_heads_options:
                for key_dim in key_dim_options:
                    for Dsize in dense_sizes:
                        for attn in attns:
                            for subj in subject_indices:
                                config_name = f"heads7_dim{key_dim}_denseSize{Dsize}_activelu, _avgFalse, _attn{attn}_avg{avgs}"
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
                                print(config_name)
                                avg_acc = np.mean(accuracies)
                                print(f"{avg_acc:.4f}")

