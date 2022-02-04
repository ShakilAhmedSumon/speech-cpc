from tensorflow import keras
from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU, Flatten, Dense, GRU, TimeDistributed, Input, Lambda
from tensorflow.keras.layers import Dot, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from keras.models import load_model



def loss_fn(y_true, y_pred):
    """
    Contrastive loss function (eq. 4 from the original article)
    # https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras
    :param y_true: labels (0, 1), where 0 means the sample was drawn from noisy distribution; 1 means the sample was
    drawn from the target distribution.
    :param y_pred: density ratio (f value from the original article)
    :return:
    """
    with tf.name_scope('custom_loss_function'):
        divident = K.sum(K.dot(y_true, y_pred), axis=1)
        divider = K.sum(y_pred, axis=1) + K.epsilon()
        l = -K.log(divident / divider)
    return l*1e4



model = load_model('models/cpc1.hdf5', custom_objects={'loss_fn': loss_fn})
model.summary()

# layer = model.get_layer('Historical_embeddings')


# print(layer)

encoder = Model(input=inputs, output=model.get_layer('Historical_embeddings'))

# X_encoded = encoder.predict(X)


# context = Model(input=inputs, output=encoded)
# X_context = context.predict(X)
