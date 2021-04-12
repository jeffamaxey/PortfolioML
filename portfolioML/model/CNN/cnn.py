""" CNN model """
import logging
import argparse
import sys
import os
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model
sys.path.append(os.path.dirname(os.path.abspath("..")))
from model.split import all_data_LSTM
from data.data_returns import read_filepath
from makedir import go_up

class MinPooling1D(MaxPooling1D):
    """
    Class for minimun pooling
    """

    def __init__(self, pool_size=2, strides=None, padding='valid', data_format='channels_last', **kwargs):
        super(MinPooling1D, self).__init__(pool_size, strides, padding, data_format, **kwargs)

    def pooling(self, x):
        """
        Compute minimun pooling
        """
        return -MaxPooling1D(self.pool_size, self.strides, self.padding, self.data_format)(-x)


def CNN_model(filters, kernel_size=(20), strides=5, activation='tanh', min_pooling=False, plt_figure=False):
    """
    CNN model with selected number of filters for the convolutional layers for classification
    task about time-series of data. Beacous of the problem one convolutional layes is enough

    The basic structure is:

    - Inputs: shape=(240,1). The shape reflect the number of close value for each istance (240)
    and the number of features (1)

    - Dropout: drop(0.1)

    - Convolution: Conv1D(filters, kernel_size=kernel_size, strides=strides, activation='tanh')(drop)
    The choise of parameters follows the "nature" of the task, the kernel_size is about one mouth
    of trading days (20) and the strides is one week (5).

    - Pooling Layers:
            -- MaxPooling: MaxPooling1D(pool_size=5, strides=1, padding='valid')(conv)
            find the max value in a pool_size of 5.

            -- If min_pooling = True
                - MinPooling: MinPooling1D(pool_size=5, strides=1, padding='valid').pooling(conv)
                find the min value in a pool_size of 5.

                - Merge: Concatenate()([min_pool, max_pool])
                Merge two pooling layers that have the same input to create a multiheaded structure

    - Dropout: drop(0.1)

    - Dense: Dense(25, activation= 'tanh')(flatten)
    Compute all the feature and the information of the previous steps

    - Output: Dense(1, activation='sigmoid'), the output is interpretated as the probability that
    the input is grater than the cross-section median.

    Parameters
    ----------
    filters: integer
        The dimensionality of the output space (the number of output filters in the convolution).
        Referances: https://keras.io/api/layers/convolution_layers/convolution1d/

    kernel_size: tuple of one integers (optional)
        Specifying the length of the 1D convolution window. Default = (20)
        Referances: https://keras.io/api/layers/convolution_layers/convolution1d/

    strides: integer
        specifying the stride length of the convolution. Default = 5
        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        Referances: https://keras.io/api/layers/convolution_layers/convolution1d/

    activation: function to use (optional)
        If you don't specify anything, no activation is applied (see keras.activations).
        Default = 'tanh'
        Referances: https://keras.io/api/layers/convolution_layers/convolution1d/


    Results
    -------
    model: tensorflow.python.keras.engine.sequential.Sequential
        tensorflow model with selected hidden layers
    """
    inputs = Input(shape=(240,1))
    drop = Dropout(0.1)(inputs)

    conv = Conv1D(filters, kernel_size=kernel_size, strides=strides, activation=activation)(drop)
    max_pool = MaxPooling1D(pool_size=5, strides=1, padding='valid')(conv)
    if min_pooling:
        min_pool = MinPooling1D(pool_size=5, strides=1, padding='valid').pooling(conv)
        merge = Concatenate()([min_pool, max_pool])
        drop = Dropout(0.4)(merge)
    else: drop = Dropout(0.4)(max_pool)

    flatten = Flatten()(drop)

    dense = Dense(25, activation= 'tanh')(flatten)
    outputs = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if plt_figure:
        plot_model(model, to_file=f'CNN:fil{filters}_kernel{kernel_size}_strides{strides}_min{min_pooling}.png',
                    show_shapes=True, show_layer_names=True)

    logging.info(model.summary())
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make CNN model for classification task to predict class label 0 or 1')
    parser.add_argument("filters", type=int, help="filters of convolutional layers")
    parser.add_argument("-kernel_size", type=tuple, default=(20), help="kernel_size, for more details see documentation")
    parser.add_argument("-strides", type=int, default=5, help="strides, for more details see documentation")
    parser.add_argument("-activation", type=str, default='tanh', help="activation, for more details see documentation")
    parser.add_argument("-min_pooling", type=bool, default=False, help="If true the structure is multiheaded")
    parser.add_argument("-plt_figure", type=bool, default=False, help="If true create png file of the model")

    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])

    #Read the data
    df_binary = go_up(2) + "/data/ReturnsBinary.csv"
    df_returns = go_up(2) + "/data/ReturnsData.csv"
    df_returns = read_filepath(df_returns)
    df_binary = read_filepath(df_binary)

    for per in range(0,10):
        model = CNN_model(args.filters, min_pooling=args.min_pooling, plt_figure=args.plt_figure)
        #Training
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, per)
        #Trainng
        es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        mc = ModelCheckpoint(f'CNN_minpool_period{per}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train ,y_train, callbacks=[es,mc],validation_split=0.2, batch_size=512, epochs=400, verbose=1)

        #Elbow curve
        plt.figure(f'CNN_minpool_Loss and Accuracy period {per}')
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Loss vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracy vs Epochs')
        plt.grid()
        plt.legend()

    plt.show()
