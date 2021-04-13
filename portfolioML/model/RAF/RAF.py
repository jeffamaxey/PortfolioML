import argparse
import logging
import requests
from statistics import median
from functools import reduce
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath("..")))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve, auc
from data.data_returns import read_filepath
from model.split import get_sequences, split_Tperiod, get_train_set, all_data_LSTM, all_data_DNN
from makedir import smart_makedir, go_up

def predictions_csv(num_period=10):
    '''
    Parameters
    ----------
    num_period : TYPE, optional
        DESCRIPTION. The default is 10.
    Returns
    -------
    None.
    -------
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
    '''

    smart_makedir(f'/results/predictions/RAF/', level_up=2)
    smart_makedir(f'/results/ROC/RAF/', level_up=2)
    
    path = go_up(level_up=2)
    path_p = path+ f'/results/predictions/RAF/'
    path_R = path + f'/results/ROC/RAF/'
    
    tpr_list = []
    aucs_list = []
    interp_fpr = np.linspace(0, 1, 10000)
    plt.figure()
    for i in range(0, num_period):
        X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, i)
        rf = RandomForestClassifier(n_estimators=1000,n_jobs=-1,min_samples_split=10,
                                    min_samples_leaf=2,random_state=10,oob_score=True, 
                                    max_depth=25, verbose=1)
        rf.fit(X_train, y_train)
        y_proba = rf.predict_proba(X_test)


        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1])
        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        tpr_list.append(interp_tpr)
        roc_auc = roc_auc_score(y_test, y_proba[:,1])
        aucs_list.append(roc_auc)

        plt.plot(fpr, tpr, label=f'per{i} (area = %0.4f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate',)
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", fontsize=12, frameon=False)
        plt.title(f'ROC CURVE')
        plt.savefig(path_R + f'ROC CURVE.png')


        y_pred_companies = [y_proba[i:87+i] for i in range(0,len(y_proba)-87+1,87)]
        dict_comp = {df_returns.columns[i]: y_pred_companies[i] for i in range(0,365)}
        df_predictions = pd.DataFrame()
        for tick in df_returns.columns:
            df_predictions[tick] = dict_comp[tick][:,0]
            df_predictions.to_csv(path_p + f'Predictions_{i}th_Period.csv')



    auc_mean = np.mean(np.array(aucs_list))
    auc_std = np.std(np.array(aucs_list))
    tpr_mean = np.mean(tpr_list, axis=0)
# -----------------------------------------------------------------------------
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
    plt.title(f'ROC Curve - mean +|- std')
    plt.savefig(path_R + f'ROC Curve - mean +|- std.png')

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RandomForestClassifier')
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

    path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    parent = os.path.abspath(os.path.join(parent_path, os.pardir))
    df_binary = parent + "/data/ReturnsBinary.csv"
    df_returns = parent + "/data/ReturnsData.csv"
    df_returns = read_filepath(df_returns)
    df_binary = read_filepath(df_binary)

    predictions_csv()

