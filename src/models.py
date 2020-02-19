import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def many2many(n_gestures=2, n_frames=300, n_features=21, rnn_units=32):
    """Model for predicting labels for a sequence of multiple gestures

    Arguments:
    n_gestures -- int, size of gesture vocabulary
    n_frames -- int, number of frames per training example
    n_features -- int, number of features
    rnn_units -- int, size of LSTM hidden state

    Note:
    Not bidirectional

    """

    inputs = tf.keras.Input(shape=(n_frames,n_features))
    # x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=False))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.LSTM(rnn_units, return_sequences=True)(inputs)
    x = layers.Dense(n_gestures, activation='softmax')(x)
    
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='many2one')

    model.summary()

    return model


def many2one_model(n_gestures=2, n_frames=35, n_features=21, rnn_units=4, bidirectional=True, n_lstm_layers=1, n_dense_layers=1, dense_size=4, recurrent_dropout=0.0):
    """Model for predicting labels for a single gesture

    Arguments:
    n_gestures -- int, size of gesture vocabulary. 2 indicates gesture/non gesture only
    n_frames -- int, number of frames per training example
    n_features -- int, number of features
    rnn_units -- int, size of LSTM hidden state
    layers -- int, number of LSTM layers

    Note:
    Bidirectional
    """

    inputs = tf.keras.Input(shape=(n_frames,n_features))
    x = inputs
    for i in range(n_lstm_layers):
        if bidirectional == True:
            if i == n_lstm_layers - 1: # check whether or not this is the last layer of LSTMs
                # if this is the last layer, the return sequences should be false
                x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=False, recurrent_dropout=recurrent_dropout))(x)
            else:
                x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, recurrent_dropout=recurrent_dropout))(x)
        else:
            if i == n_lstm_layers - 1: # check whether or not this is the last layer of LSTMs
                x = layers.LSTM(rnn_units, return_sequences=False, stateful=False, recurrent_dropout=recurrent_dropout)(x)
            else:
                x = layers.LSTM(rnn_units, return_sequences=True, stateful=False, recurrent_dropout=recurrent_dropout)(x)
    
    for i in range(n_dense_layers - 1):
        x = layers.Dense(dense_size, activation='relu')(x)

    
    x = layers.Dense(n_gestures, activation='softmax')(x)
    
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='many2one')

    model.summary()

    return model

    
def plt_metric(history, metric='loss'):
    """plots metrics from the history of a model
    
    Arguments:
    history -- history of a keras model
    metric -- str, metric to be plotted
    
    """

    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plt_pred(y, pred):
    """Plots truth labels vs predicted labels for an example"""
    labels = np.argmax(np.squeeze(pred), axis=-1)
    plt.plot(labels)
    plt.plot(y)
    plt.title('predicted vs labels')
    plt.ylabel('label')
    plt.xlabel('time step')
    plt.legend(['predicted', 'labels'], loc='upper left')
    plt.show()

