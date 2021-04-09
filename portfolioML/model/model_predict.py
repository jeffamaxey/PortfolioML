""" Prediction with DNN model """
import numpy as np
import pandas as pd
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.models import load_model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from dnn import all_data_DNN, all_data_LSTM
from portfolioML.data.data_returns import read_filepath

def plot_roc(model, periods):
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
        File path of the model, it must have in the name 'periond#' (for istance 'period0') to
        take in mind the numer of perion over wich the model is trained

    periods: integer 
        Numer of model that are taken in order to compute plot and final values

    """
    tpr_list = []
    aucs_list = []
    interp_fpr = np.linspace(0, 1, 10000)
    for per in range(0,periods):
        #Splitting data set for each period
        X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, per)
        
        model = load_model(f'../model/DNN_bestia_adam_period{per}y.h5')

        #ROC curve  
        probas = model.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, probas[:,0])

        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        tpr_list.append(interp_tpr)

        roc_auc = roc_auc_score(y_test, probas[:,0], average=None)
        aucs_list.append(roc_auc)

        plt.figure('ROC CURVES')
        plt.plot(fpr, tpr, label=f'per{per} (area = %0.4f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel('False Positive Rate',)
        plt.ylabel('True Positive Rate')
        plt.title('ROC CURVE')
        plt.legend(loc="lower right", fontsize=12, frameon=False)

    auc_mean = np.mean(np.array(aucs_list))
    auc_std = np.std(np.array(aucs_list))

    tpr_mean = np.mean(tpr_list, axis=0)
    
    plt.figure('ROC CURVE - mean pm std')
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
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction and compare traned model')
    parser.add_argument('returns_file', type=str, help='Path to the returns input data')
    parser.add_argument('binary_file', type=str, help='Path to the binary target data')
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
    df_returns = read_filepath(args.returns_file)
    df_binary = read_filepath(args.binary_file)
    
    plt.figure()
    plt.plot(df_returns.XRX)

    tpr_list = []
    fpr_list = []
    aucs_list = []

    interp_fpr = np.linspace(0, 1, 10000)
    for per in range(0,10):
        #Splitting data set for each period
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, per)
        
        model = load_model(f'../model/CNN_dense3_period{per}.h5')

        #ROC curve  
        probas = model.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, probas[:,0])

        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        tpr_list.append(interp_tpr)

        roc_auc = roc_auc_score(y_test, probas[:,0], average=None)
        aucs_list.append(roc_auc)

        plt.figure('ROC CURVES')
        plt.plot(fpr, tpr, label=f'per{per} (area = %0.4f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel('False Positive Rate',)
        plt.ylabel('True Positive Rate')
        plt.title('ROC CURVE')
        plt.legend(loc="lower right", fontsize=12, frameon=False)

    auc_mean = np.mean(np.array(aucs_list))
    auc_std = np.std(np.array(aucs_list))

    tpr_mean = np.mean(tpr_list, axis=0)
    
    plt.figure('ROC CURVE - mean pm std')
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
    plt.show()