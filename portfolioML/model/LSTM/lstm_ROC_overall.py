""" Prediction with LSTM model """
import numpy as np
import pandas as pd
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import sys
import os
import shutil

def go_up(level_up):
    if level_up == 0:
        path = os.getcwd()
    if level_up == 1:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if level_up == 2:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    if level_up == 3:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    if level_up == 4:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def smart_makedir(name_dir, level_up=0):
    '''
    Easy way to create a folder. You can go up from the current path up to 4 times.

    Parameters
    ----------
    name_dir : str
        From level you have set, complete path of the directory you want to create
    level_up : TYPE, optional
        How many step up you want to do from the current path. The default is 0.

    Returns
    -------
    None.

    '''

    if level_up == 0:
        path = go_up(0) + "/" + name_dir
    if level_up == 1:
        path = go_up(1) + "/" + name_dir
    if level_up == 2:
        path = go_up(2) + "/" + name_dir
    if level_up == 3:
        path = go_up(3) + "/" + name_dir
    if level_up == 4:
        path = go_up(4) + "/" + name_dir

    if os.path.exists(path):
        logging.info(f"Path '{path}' already exists, it will be overwritten \n")
        # Remove all the files in case they already exist
        shutil.rmtree(path)
    os.makedirs(path)
    logging.info(f"Successfully created the directory '{path}' \n")

def read_filepath(file_path):
    """
    Read and compute basic informations about a data set in csv.

    Parameters
    ----------
    file_path: str
        Path to the csv file

    Returns
    -------
    df: pandas dataframe
        Pandas Dataframe of the read file
    """
    name_file = file_path.split('/')[-1]
    extention = name_file.split('.')[-1]

    try:
        if extention != 'csv':
            raise NameError('The file is not a csv one')
    except NameError as ne:
        logging.error(ne)
        exit()

    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.drop(['Days'], axis=1)

    # logging.info('DATA INFO, attributes: %s', df.columns)
    # logging.info('DATA INFO, shape: %s', df.shape)

    # total_data = df.shape[0]*df.shape[1]
    # total_missing = pd.DataFrame.isnull(df).sum().sum()

    # logging.info(f'DATA QUALITY, missing values: {total_missing/total_data:.2%}')

    return df

def split_Tperiod(df_returns, df_binary, len_period=1308, len_test=327):
    """
    Split the entire dataframe in study period T, each of them having len_period
    elements of which len_test account for the trading set. To generate all the
    periods, a rolling window of len_period lenght is moved along the entire
    dataset in len_test steps.


    Parameters
    ----------
    df_returns: pandas dataframe
        Pandas dataframe of returns.

    df_binary: pandas dataframe
        Pandas dataframe of binary targets.

    len_period: integer(optional)
        Lenght of the study period

    len_test: integer(optional)
        Lenght of the trading set.
    Results
    -------
    periods: list of pandas dataframe
        List of pandas dataframe of all periods of lenght len_period.
    """

    len_total_leave = len(df_returns)-len_period #ho solo chiamato come unica variabile quella cosa che c'era nel for, il nome Ã¨ da rivedere
    periods_ret = [(df_returns[i:len_period+i]) for i in range(0, len_total_leave+1, len_test)]
    periods_bin = [(df_binary[i:len_period+i]) for i in range(0, len_total_leave+1, len_test)] # questa mancava

    return periods_ret, periods_bin


def get_sequences(returns, targets, n_steps=240):
    """
    Returns the  sequences of inputs and targets for classification task (not already
    ready for the LSTM).


    Parameters
    ----------
    returns: pandas dataframe
        pandas dataframe of time-series data of returns to split.

    targets: pandas dataframe
        pandas dataframe of time-series data of target to split. It must have the same length of returns

    n_steps: integer(optional)
        number of time steps for each istance. Default = 100

    Results
    -------
    X: list
        Array of the input set, its shape is (len(sequences)-n_steps, n_steps)

    y: list
        Array of the input target, its shape is (len(sequences)-n_steps, 1)
    """
    try:
        returns = returns.to_numpy()
        targets = targets.to_numpy()
    except AttributeError:
        pass

    X = [returns[i:i+n_steps] for i in range(len(returns)-n_steps)]
    y = [targets[i+n_steps] for i in range(len(targets)-n_steps)]

    return X, y


def get_train_set(df_returns, df_binary):
    """
    Return the train set for the LSTM.
    The argumets are the returns dataframe and the binary dataframe. The function compute respectively
    the X_train and the y_train for classification task, stacking sequences of different companies
    one over another.

    Parameters
    ----------
    df_returns: pandas dataframe, numpy array
        Dataframe of returns

    df_binary: pandas dataframe, numpy array
        Datframe of binary target associated to data returns. It has the same shape of df_returns

    Returns
    -------
    list_tot_X: numpy array
        Array of input data for LSTM

    list_tot_y: numpy array
        Array of input target class for LSTM
    """

    list_tot_X = []
    list_tot_y = []

    if (str(type(df_returns)) != "pandas.core.frame.DataFrame"):
        df_returns = pd.DataFrame(df_returns)
    if (str(type(df_binary)) != "pandas.core.frame.DataFrame"):
        df_binary = pd.DataFrame(df_binary)

    for comp in df_returns.columns:
        X_train, y_train = get_sequences(df_returns[comp], df_binary[comp])
        list_tot_X.append(X_train)
        list_tot_y.append(y_train)

    list_tot_X = np.array(list_tot_X)
    list_tot_y = np.array(list_tot_y)

    list_tot_X = np.reshape(list_tot_X,(list_tot_X.shape[0]*list_tot_X.shape[1],list_tot_X.shape[2]))
    list_tot_y = np.reshape(list_tot_y,(list_tot_y.shape[0]*list_tot_y.shape[1]))

    return list_tot_X, list_tot_y

def all_data_LSTM(df_returns, df_binary, period, len_train=981):
    """
    Function that create the right input for the LSTM algorithm.
    X_train and X_test are normalized. X_train is reshaped.

    Parameters
    ----------
    df_returns : pandas dataframe
        Pandas dataframe of returns.
    df_binary : pandas dataframe
        Pandas dataframe of returns..
    period : int
        Period over which you wanto to create the input for the LSTM.
    len_train : int, optional
        Lenght of the training set. The default is 981.
    len_test : int, optional
        Lenght of the trading set. The default is 327.

    Returns
    -------
    X_train : numpy array

    y_train : numpy array

    X_test : numpy array

    y_test : numpy array

    """
    scaler = StandardScaler()
    periods_returns, periods_binary = split_Tperiod(df_returns, df_binary)

    T1_input = periods_returns[period]
    T1_target = periods_binary[period]

    T1_input[:len_train] = scaler.fit_transform(T1_input[:len_train])
    X_input_train, y_input_train = T1_input[:len_train], T1_target[:len_train]

    T1_input[len_train:] = scaler.fit_transform(T1_input[len_train:])
    X_test, y_test = T1_input[len_train:], T1_target[len_train:]

    X_train, y_train = get_train_set(X_input_train, y_input_train)
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test, y_test = get_train_set(X_test, y_test)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, X_test, y_test

def plot_roc(algorithm, name_model, num_periods):
    """
    Plot roc curve with mean and standard deviation of the area under the curve (auc) of a
    trained model.
    Each model is trained over several study period so its name contain this information,
    for semplicity put this information in this way: '..._periond#' (for istance '_period0').
    Nember of period running over several argumenti setting in 'periods' argument.
    So before running thi function carefully checking of the folder is suggest to avoid
    problems.

    Tecnical aspects: because of each model has got different tpr and fps, an interpoletion
    of this values is used in ordet to have the same leght for each model.

    Parameters
    ----------
    model: string
        File path of the model, it must have in the name 'periond' (for istance 'period0') to
        take in mind the numer of perion over wich the model is trained

    periods: integer
        Numer of model that are taken in order to compute plot and final values

    """

    logging.info('----- I am creating ROC curve png files -----')

    parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    tpr_list = []
    aucs_list = []
    interp_fpr = np.linspace(0, 1, 10000)

    smart_makedir(f'/ROC/{algorithm}/{name_model}/')

    path = os.getcwd() + f'/ROC/{algorithm}/{name_model}/'

    plt.figure()
    for per in range(0,num_periods):
        logging.info(f'Creating ROC for period {per}')
        if (algorithm == 'LSTM') or (algorithm == 'CNN'):
            #Splitting data set for each period
            X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, per)
        else:
            X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, per)

        model = load_model(parent_path + f'/model/{algorithm}/{name_model}/{name_model}_period{per}.h5')

        #ROC curve
        probas = model.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, probas[:,0])

        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        tpr_list.append(interp_tpr)

        roc_auc = roc_auc_score(y_test, probas[:,0], average=None)
        aucs_list.append(roc_auc)


        plt.plot(fpr, tpr, label=f'per{per} (area = %0.4f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate',)
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize=12, frameon=False)
    plt.title(f'ROC CURVE {name_model}')
    plt.savefig(path + f'ROC CURVE {name_model}.png')

    auc_mean = np.mean(np.array(aucs_list))
    auc_std = np.std(np.array(aucs_list))

    tpr_mean = np.mean(tpr_list, axis=0)

    plt.figure()
    plt.plot(interp_fpr, tpr_mean, color='b',
          label=f'Mean ROC (AUC = {auc_mean:.4f} $\pm$ {auc_std:.4f})',
          lw=1, alpha=.8)

    tpr_std = np.std(tpr_list, axis=0)
    tprs_upper = np.minimum(tpr_mean + tpr_std, 1)
    tprs_lower = np.maximum(tpr_mean - tpr_std, 0)
    plt.fill_between(interp_fpr, tprs_lower, tprs_upper, color='blue', alpha=.2,
                  label=r'$\pm$ 1 std. dev.')
    plt.xlabel('False Positive Rate',)
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize=12, frameon=False)
    plt.title(f'ROC Curve {name_model} - mean +|- std')
    plt.savefig(path + f'ROC Curve {name_model} - mean +|- std.png')

def predictions_csv(algorithm, model_name, num_periods):
    '''


    Parameters
    ----------
    num_periods : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    '''

    logging.info('----- I am creating predictions csv file -----')

    smart_makedir(f'/predictions/{algorithm}/{model_name}/')

    path = os.getcwd() + f'/predictions/{algorithm}/{model_name}/'
    parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    for i in range(num_periods):
        model = load_model(parent_path + f'/model/{algorithm}/{model_name}/{model_name}_period{i}.h5')
        if (algorithm == 'LSTM') or (algorithm == 'CNN'):
            #Splitting data set for each period
            X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)
        else:
            X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, i)

        y_pred = model.predict(X_test)
        y_pred_companies = [y_pred[i:87+i] for i in range(0,len(y_pred)-87+1,87)]
        dict_comp = {df_returns.columns[i]: y_pred_companies[i] for i in range(0,365)}
        df_predictions = pd.DataFrame()
        for tick in df_returns.columns:
            df_predictions[tick] = dict_comp[tick][:,0]
            df_predictions.to_csv(path + f'{model_name}_Predictions_{i}th_Period.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction and compare traned model')
    parser.add_argument('algorithm', type=str, help='CNN. LSTM or RAF')
    parser.add_argument('model_name', type=str, help='Select the particular model trained')
    parser.add_argument('num_periods', type=int, help="Number of period over which returns have to be calculated ")
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])
    pd.options.mode.chained_assignment = None

    #Read the data
    path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    df_binary = parent_path + "/data/ReturnsBinary.csv"
    df_returns = parent_path + "/data/ReturnsData.csv"
    df_returns = read_filepath(df_returns)
    df_binary = read_filepath(df_binary)


    plot_roc(args.algorithm, args.model_name, args.num_periods)
    predictions_csv(args.algorithm, args.model_name, args.num_periods)
