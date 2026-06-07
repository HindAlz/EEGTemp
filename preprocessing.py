import mat73
import numpy as np
import pickle
import random
import scipy.signal as sp

mat = mat73.loadmat('/mnt/c/Users/Hzaab/Downloads/MIT_img_bl2anddata.mat')
proc_data = mat["data"]

#outputs of the code
X_data_nt_st = []
Y_data_nt_st = []

X_data_nt_ct = []
Y_data_nt_ct = []

X_data_ct_st = []
Y_data_ct_st = []

X_data_four = []
Y_data_four = []

X_data_three = []
Y_data_three = []

X_data_mi_nonMi = []
Y_data_mi_nonMi = []

def data_gen(data1, data2, X_Output, Y_Output):
    X_data_nt = np.concatenate([data1, data2], axis=0)
    y_data_nt = np.concatenate([np.full(len(data1), 1), np.full(len(data2), 0)])
    X_Output.append(X_data_nt)
    Y_Output.append(y_data_nt)
import numpy as np
import random

def average_random_trials(trial_list, num_trials_to_avg=5):
    new_list = []
    for _ in range(len(trial_list)):
        sampled = random.choices(trial_list, k=num_trials_to_avg)
        avg_trial = np.mean(sampled, axis=0)
        new_list.append(avg_trial)
    return new_list

# Assuming proc_data is already loaded
for subj in range(len(proc_data)):  # iterate over all subjects
    nt_data = []  # natural
    st_data = []  # super
    ct_data = []  # combined
    rest_data_total = []
    rest_data_nt = []
    rest_data_st = []
    rest_data_ct = []

    for i in range(3):  # iterate over all 3 sessions
        cur_sesh = proc_data[subj][i]  # shape: (60, 5000, num_trials)

        for trial in range(cur_sesh.shape[2]):
            # Extract rest and MI parts
            rest_data = cur_sesh[:, 0:2000, trial][:, ::5]  # downsampled
            Mi = cur_sesh[:, 2000:5000, trial][:, ::5]      # downsampled
            Mi = sp.resample(Mi, 400, axis=1)

            rest_data = np.expand_dims(rest_data, axis=0)  # shape: (1, chans, time)
            Mi = np.expand_dims(Mi, axis=0)

            rest_data_total.append(rest_data)

            if i == 0:
                nt_data.append(Mi)
                rest_data_nt.append(rest_data)
            elif i == 1:
                st_data.append(Mi)
                rest_data_st.append(rest_data)
            elif i == 2:
                ct_data.append(Mi)
                rest_data_ct.append(rest_data)

    # Average 5 random trials and overwrite the original lists
    nt_data = average_random_trials(nt_data)
    st_data = average_random_trials(st_data)
    ct_data = average_random_trials(ct_data)

    #rest_data_nt = average_random_trials(rest_data_nt)
    #rest_data_st = average_random_trials(rest_data_st)
    #rest_data_ct = average_random_trials(rest_data_ct)
    rest_data_total = average_random_trials(rest_data_total)

    #keep only a third of the rest data
    rest_data_total = random.sample(rest_data_total, len(rest_data_total) // 3)
    #print("nt_data count:", len(nt_data))
    #print("rest_data count:", len(rest_data))
    #print("total samples:", len(nt_data) + len(rest_data))

    '''
    #nt vs st
    data_gen(nt_data, st_data, X_data_nt_st, Y_data_nt_st)
    #nt vs ct
    data_gen(nt_data, ct_data, X_data_nt_ct, Y_data_nt_ct)
    #st vs ct
    data_gen(ct_data, st_data, X_data_ct_st, Y_data_ct_st)

    #st vs ct vs nt vs rest
    X_data = np.concatenate([ct_data, nt_data, st_data, rest_data_total], axis=0)
    y_data = np.concatenate([np.full(len(ct_data), 0), np.full(len(nt_data), 1), np.full(len(st_data), 2), np.full(len(rest_data_total), 3)])
    X_data_four.append(X_data)
    Y_data_four.append(y_data)


    # st vs ct vs nt
    X_data = np.concatenate([ct_data, nt_data, st_data], axis=0)
    y_data = np.concatenate([np.full(len(ct_data), 0), np.full(len(nt_data), 1), np.full(len(st_data), 2)])
    X_data_three.append(X_data)
    Y_data_three.append(y_data)

'''
    #mi (st+ct) vs non mi (rest+nt)
    X_data = np.concatenate([ct_data, st_data, nt_data, rest_data_total], axis=0)
    mi=len(ct_data)+len(st_data)
    nonMi=len(nt_data)+len(rest_data_total)
    y_data = np.concatenate([np.full(mi, 0), np.full(nonMi, 1)])
    X_data_mi_nonMi.append(X_data)
    Y_data_mi_nonMi.append(y_data)
'''

with open('processed_data_nt_st.pkl', 'wb') as f:
    pickle.dump((X_data_nt_st, Y_data_nt_st), f)

with open('processed_data_nt_ct.pkl', 'wb') as f:
    pickle.dump((X_data_nt_ct, Y_data_nt_ct), f)

with open('processed_data_ct_st.pkl', 'wb') as f:
    pickle.dump((X_data_ct_st, Y_data_ct_st), f)

with open('processed_data_four_nonRes.pkl', 'wb') as f:
    pickle.dump((X_data_four, Y_data_four), f)
'''

with open('processed_data_mi_nonMi_Avg.pkl', 'wb') as f:
    pickle.dump((X_data_mi_nonMi, Y_data_mi_nonMi), f)

'''
with open('processed_data_three.pkl', 'wb') as f:
    pickle.dump((X_data_three, Y_data_three), f)

'''

#number of subjs