import mat73
import pickle
import scipy.signal as sp
import numpy as np
import random

mat = mat73.loadmat('/mnt/c/Users/Hzaab/Downloads/MIT_img_bl2anddata.mat')
proc_data = mat["data"]

#outputs of the code
X_data_mi_nonMi = []
Y_data_mi_nonMi = []

def average2(trials):
    new = []
    for i in range(len(trials)):
        sampled = random.choices(trials, k=5)
        avg_trial = np.mean(sampled, axis=0)
        new.append(avg_trial)
    return new

for subj in range(len(proc_data)):
    nt_data = []
    st_data = []
    ct_data = []
    rest_data_total = []

    for i in range(3):
        cur_sesh = proc_data[subj][i]

        for trial in range(cur_sesh.shape[2]):
            rest_data = cur_sesh[:, 0:2000, trial][:, ::5]
            Mi = cur_sesh[:, 2000:5000, trial][:, ::5]
            Mi = sp.resample(Mi, 400, axis=1)

            rest_data = np.expand_dims(rest_data, axis=0)
            Mi = np.expand_dims(Mi, axis=0)

            rest_data_total.append(rest_data)

            if i == 0:
                nt_data.append(Mi)
            elif i == 1:
                st_data.append(Mi)
            elif i == 2:
                ct_data.append(Mi)

    nt_data = average2(nt_data)
    st_data = average2(st_data)
    ct_data = average2(ct_data)
    rest_data_total = average2(rest_data_total)
    rest_data_total = random.sample(rest_data_total, len(rest_data_total) // 3)

    #mi (st+ct) vs non mi (rest+nt)
    X_data = np.concatenate([ct_data, st_data, nt_data, rest_data_total], axis=0)
    mi=len(ct_data)+len(st_data)
    nonMi=len(nt_data)+len(rest_data_total)
    y_data = np.concatenate([np.full(mi, 0), np.full(nonMi, 1)])
    X_data_mi_nonMi.append(X_data)
    Y_data_mi_nonMi.append(y_data)

with open('processed_data_mi_nonMi_Avg.pkl', 'wb') as f:
    pickle.dump((X_data_mi_nonMi, Y_data_mi_nonMi), f)
