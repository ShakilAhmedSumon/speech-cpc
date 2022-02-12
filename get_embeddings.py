from posixpath import dirname
from random import sample
from unicodedata import category
import librosa
import os
import json
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.layers import Dot, Lambda
from tensorflow.keras.models import Model


DATASET_PATH = "VALIDATION_DATA"
SAVED_EMBEDDED_PATH = "embeddings"
SAMPLES_TO_CONSIDER = 20480

def get_embeddings(dataset_path):
    print("Getting embeddings...")
    # loop through all the sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        print("Processing dir: {}".format(dirpath))
        if dirpath is not dataset_path:
            category = dirpath.split('\\')[-1]         # dirpath.split('/')[-1] for linux
            arr = []
            for f in filenames:
                
                # constract the filepath
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, 20480)

                # ensure the audio is atleast 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    signal = signal.reshape(-1, 4096)
                    signal = signal.reshape(5, 4096, 1)
                    arr.append(signal)

            
            # print(np.asarray(arr).shape)
            arr = np.asarray(arr)
            emb = get_embedding(arr)
            # print(emb.shape)

            if not os.path.exists(SAVED_EMBEDDED_PATH + '/' + DATASET_PATH):
                os.makedirs(SAVED_EMBEDDED_PATH + '/' + DATASET_PATH)
            
            np.save(SAVED_EMBEDDED_PATH + '/' + DATASET_PATH + '/' + str(category), emb)
            

 

def loss_fn(y_true, y_pred):

    with tf.name_scope('custom_loss_function'):
        divident = K.sum(K.dot(y_true, y_pred), axis=1)
        divider = K.sum(y_pred, axis=1) + K.epsilon()
        l = -K.log(divident / divider)
    return l*1e4




def get_embedding(signal):
    encoder = Model(inputs=model.get_layer('context_data').output, outputs=model.get_layer('Historical_embeddings').output)
    # encoder.summary()
    embedding = encoder.predict(signal)

    return embedding




if __name__=="__main__":
    model = load_model('models/cpc.hdf5', custom_objects={'loss_fn': loss_fn})
    get_embeddings(DATASET_PATH)

    # signal, sr = librosa.load('speech_data/eight/1ecfb537_nohash_3.wav', 20480)
    # print("Before Reshaping: ", signal.shape)
    # signal = signal.reshape(-1, 4096, 1)
    # # signal = signal.reshape(1, 5, 4096, 1)
    # print("After Reshaping: ", signal.shape)

    