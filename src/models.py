import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def many2many_model(n_gestures=2, n_frames=300, n_features=21, rnn_units=32):
    """Model for predicting labels for a sequence of multiple gestures

    Arguments:
    n_gestures -- int, size of gesture vocabulary
    n_frames -- int, number of frames per training example
    n_features -- int, number of features
    rnn_units -- int, size of LSTM hidden state

    Note:
    Not bidirectional

    """

    model = tf.keras.Sequential()
    model.add(layers.LSTM(rnn_units, return_sequences=True, stateful=False, input_length=n_frames, input_dim=n_features))
    model.add(layers.TimeDistributed(layers.Dense(n_gestures, activation='softmax')))
    model.summary()

    return model


def many2one_model(n_gestures=2, n_frames=120, n_features=21,  rnn_units=64):
    """Model for predicting labels for a single gesture

    Arguments:
    n_gestures -- int, size of gesture vocabulary. 2 indicates gesture/non gesture only
    n_frames -- int, number of frames per training example
    n_features -- int, number of features
    rnn_units -- int, size of LSTM hidden state

    Note:
    Bidirectional
    """

    inputs = tf.keras.Input(shape=(n_frames,n_features))

    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=False, stateful=False))(inputs)
    x = layers.Dense(n_gestures, activation='softmax')(x)
    
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='many2one')

    model.summary()

    return model

    
