import pickle
import numpy as np

with open('processed_data_nt.pkl', 'rb') as f:
    subjects_X_data_nt, subject_y_data_nt = pickle.load(f)

with open('processed_data_st.pkl', 'rb') as f:
    subjects_X_data_st, subject_y_data_st = pickle.load(f)

with open('processed_data_ct.pkl', 'rb') as f:
    subjects_X_data_ct, subject_y_data_ct = pickle.load(f)

print(np.concatenate(subjects_X_data_nt, axis=0).shape)