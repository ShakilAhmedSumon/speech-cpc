import preprocess as pp
import numpy as np
from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU, Flatten, Dense, GRU, TimeDistributed, Input, Lambda
from tensorflow.keras.layers import Dot, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import os
import datetime
from sklearn import preprocessing

TRAIN_DATA_PATH = "C:\\Users\\Niyaz B. Hashem\\Desktop\\speech-cpc\\embeddings\\DATA\\"
VAL_DATA_PATH = "C:\\Users\\Niyaz B. Hashem\\Desktop\\speech-cpc\\embeddings\\VALIDATION_DATA\\"
CLASS_NUM = 12

X, y = pp.load_process_data(TRAIN_DATA_PATH)
X_valid, y_valid = pp.load_process_data(VAL_DATA_PATH)

perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]


valid_perm = np.random.permutation(len(X_valid))
X_valid = X_valid[valid_perm]
y_valid = y_valid[valid_perm]

print(f"X_train: {X.shape}, y_train: {y.shape}")
print(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
y_cat = label_encoder.fit_transform(y)
y_valid_cat = label_encoder.fit_transform(y_valid)

y_onehot = tf.keras.utils.to_categorical(y_cat, num_classes=CLASS_NUM)
y_valid_onehot = tf.keras.utils.to_categorical(y_valid_cat, num_classes=CLASS_NUM)


input_model = Input(shape=[5, 512])
x = Conv1D(filters=8, groups=4, strides=2, kernel_size=8, padding = 'same')(input_model)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Conv1D(filters=4, groups = 4, kernel_size=3)(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(units=CLASS_NUM, activation='softmax')(x)


grouped_model = Model(input_model, x)
grouped_model.summary()



params = {'model_name': 'grouped_model'}
params.update({'checkpointer': {'verbose': 1,
                                'save_best_only': True,
                                'mode': 'max',
                                'monitor': 'accuracy'}})
checkpointer = ModelCheckpoint(filepath=os.path.join('models/', params.get('model_name')+'.hdf5'),
                                   **params['checkpointer'])
tensorboard = TensorBoard(log_dir='./logs', write_graph=True)
                
callbacks = [tensorboard, checkpointer]

grouped_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics='accuracy')
grouped_model.fit(X, y_onehot, epochs=100, batch_size=32, validation_data=(X_valid, y_valid_onehot), callbacks=callbacks)
