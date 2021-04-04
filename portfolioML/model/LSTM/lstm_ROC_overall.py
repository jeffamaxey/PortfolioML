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
from lstm import all_data_LSTM
from portfolioML.data.data_returns import read_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make DNN for classification task to predicti class label 0 or 1')
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

    tpr_list = []
    fpr_list = []
    aucs_list = []

    interp_fpr = np.linspace(0, 1, 10000)
    for i in range(0,10):
        #Splitting data set for each period
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)

        model = load_model(f'/home/danielemdn/Scrivania/Modello1/1/LSTM_{i}_period.h5')

        #ROC curve
        probas = model.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, probas[:,0])

        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        tpr_list.append(interp_tpr)

        roc_auc = roc_auc_score(y_test, probas[:,0], average=None)
        aucs_list.append(roc_auc)

        plt.figure('ROC CURVES')
        plt.plot(fpr, tpr, label=f'hid3-per{i} (area = %0.4f)' % (roc_auc))
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
