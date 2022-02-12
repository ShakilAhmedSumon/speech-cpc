import numpy as np
import os


KNOWN_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"]

def load_process_data(DATA_PATH):

    known_npy = np.empty([0, 5, 512])
    known_label = np.empty([0,])
    unknown_npy = np.empty([0, 5, 512])

    for file in os.listdir(DATA_PATH):
        if file.endswith(".npy"):
            file_npy = np.load(DATA_PATH + file)
            if file.split('.')[0] in KNOWN_CLASSES:
                known_npy = np.concatenate((known_npy, file_npy), axis=0)
                label = np.empty(file_npy.shape[0], dtype=object)
                label.fill(file.split('.')[0])
                known_label = np.concatenate((known_label, label), axis=0)
                print(f"Known class: {file.split('.')[0]} - Shape: {file_npy.shape} - Label: {label.shape}")
            else:
                unknown_npy = np.concatenate((unknown_npy, file_npy), axis=0)
                print(f"Unknown class: {file.split('.')[0]} - Shape: {file_npy.shape}")
                
    unknown_label = np.empty(unknown_npy.shape[0], dtype=object)
    unknown_label.fill('unknown')

    print("-"*50)
    print(f"Known Class: {known_npy.shape}, Known Label: {known_label.shape}")
    print(f"Unknown Class: {unknown_npy.shape}, Unknown Label: {unknown_label.shape}")
    print("-"*50)

    x = np.concatenate((known_npy, unknown_npy), axis=0)
    y = np.concatenate((known_label, unknown_label), axis=0)
    # print(f"X: {x.shape}, y: {y.shape}")
 
    # known_perm, unknown_perm = np.random.permutation(len(known_npy)), np.random.permutation(len(unknown_npy))
    # known_npy, unknown_npy = known_npy[known_perm], unknown_npy[unknown_perm]
    # known_label, unknown_label = known_label[known_perm], unknown_label[unknown_perm]

    return x, y




