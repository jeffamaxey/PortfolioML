"""DNN model"""
import logging
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
sys.path.append(os.path.dirname(os.path.abspath("..")))
from model.split import all_data_DNN
from data.data_returns import read_filepath
from makedir import go_up

def DNN_model(nodes_args, hidden=None , activation='tanh', loss='binary_crossentropy', optimizer='adam'):
    """
    DNN model with selected number of hidden layer for classification task.
    For more details about the model see the reference
    The model is maded by:

    - Input: shape = (feature), features are the numer of values taken from the past,
    follow the leterature the default is 31.

    - Hidden Layers: Dense(feature, activation=activation), sequential hidden layers full-connected
    with different nodes. If hiddin is an integer the number of nodes for each layer
    follow a descrescent way from 31 to 5.
    Note that the actual values of the nodes is determine by np.linspace(feature,5,hidden).

    - Output: Dense(1, activation='sigmoid'), the output is interpretated as the probability that
    the input is grater than the cross-section median.

    Note that a suitable Dropout layers fill between the layers described above. Parameters of this
    layers has been choosen following a "try and error" way to minimaze the shape of loss fuction
    (future version of this code will have the possibility to set this parameters).

    Reference: "doi:10.1016/j.ejor.2016.10.031"

    Parameters
    ----------

    nodes_args: list of integer
        Number of nodes for each layers.

    hidden: integer(optional), default = None
        Number of hidden layers, the actual values of the nodes are fixed in descrescent way
        from 3 to 5 through the np.linspace function (np.linspace(31,5,hidden)).
        Follow some useful example:
        - 3: [31,18,5]
        - 4: [31,22,13,5]
        - 5: [31,24,18,11,5]
        - 6: [31,25,20,15,10,5]

    activation: string(optional)
        Activation function to use of hidden layers, default='tanh'
        Reference: https://keras.io/api/layers/core_layers/dense/

    loss: string (name of objective function) (optional)
        objective function or tf.keras.losses.Loss instance. See tf.keras.losses.
        Loss fuction, defaul=binary_crossentropy'
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model

    optimater: string(optional)
        String (name of optimizer) or optimizer instance. See tf.keras.optimizers., default='adam'
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model


    Returns
    -------
    model: tensorflow.python.keras.engine.sequential.Sequential
        tensorflow model with selected hidden layers

    """
    model = Sequential()

    model.add(Input(shape=(31)))
    model.add(Dropout(0.1))

    if hidden is not None:
        logging.info("Nember of layers is determined by argument hidden")
        nodes = [int(i) for i in np.linspace(31,5,hidden)]
    else:
        nodes = nodes_args

    for nod in nodes:
        model.add(Dense(nod, activation=activation))
        model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    logging.info(model.summary())
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make DNN for classification task to prediction class label 0 or 1')
    parser.add_argument('num_periods', type=int, help='Number of periods you want to train')
    parser.add_argument('model_name', type=str, help='Choose the name of the model')
    parser.add_argument("nodes", type=int, nargs='+',
                        help='Number of nodes in each layers of DNN, see documentation')
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

    smart_makedir(args.model_name)
    # losses = smart_makedir(args.model_name + "/losses")
    accuracies = smart_makedir(args.model_name + "/accuracies_losses")

    for per in range(args.num_periods):
        model = DNN_model(args.nodes, optimizer='adam')
        #Splitting data for each period
        X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, per)
        #Trainng
        es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        mc = ModelCheckpoint(f'{args.model_name}/{args.model_name}_period{per}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train ,y_train, callbacks=[es,mc],validation_split=0.2, batch_size=256, epochs=1, verbose=1)

        #Elbow curve
        plt.figure(f'Loss and Accuracy Period {per}', figsize=[20.0,10.0])
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Losses vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracies vs Epochs')
        plt.grid()
        plt.legend()
        plt.savefig(os.getcwd() + f'/{args.model_name}/accuracies_losses/accuracies_{per}.png')

    plt.show()

    with open(f"{args.model_name}/{args.model_name}_specifics.txt", 'w', encoding='utf-8') as file:
        file.write(f'\n Model Name: {args.model_name} \n Number of periods: {args.num_periods} \n Number of nodes: {args.nodes} \n \n')
        model.summary(print_fn=lambda x: file.write(x + '\n'))
