from keras.models import load_model
from sklearn.metrics import roc_curve, auc, roc_auc_score
import sys
import os
import argparse
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath("..")))
from portfolioML.data.data_returns import read_filepath
from lstm import all_data_LSTM

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Comparision between different models via ROC Curve')
    parser.add_argument('returns_file', type=str, help='Path to the returns input data')
    parser.add_argument('binary_file', type=str, help='Path to the binary target data')
    parser.add_argument('num_periods', type=int, help='Number of periods you want to train')
    args = parser.parse_args()

    df_returns = read_filepath(args.returns_file)
    df_binary = read_filepath(args.binary_file)

    for i in range(args.num_periods):
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)
        model1 = load_model(f"/home/danielemdn/Scrivania/LSTM(25 10 400 640 True, Dropout=0.1 x 2)/LSTM_{i}_period.h5")
        model2 = load_model(f"/home/danielemdn/Scrivania/LSTM(25 10 400 896 False) Dropout=0.1 x2/LSTM_{i}_period.h5.h5")

        probas_1 = model1.predict(X_test)
        probas_2 = model2.predict(X_test)

        fpr1, tpr1, thresholds = roc_curve(y_test, probas_1[:, 0])
        fpr2, tpr2, thresholds2 = roc_curve(y_test, probas_2[:, 0])

        roc_auc1 = roc_auc_score(y_test, probas_1[:,0], average=None)
        roc_auc2 = roc_auc_score(y_test, probas_2[:,0], average=None)

        plt.figure(f'ROC CURVE PERIOD {i}')
        plt.plot(fpr1, tpr1, label='Model 1 (area = %0.4f)' % (roc_auc1))
        plt.plot(fpr2, tpr2, label='Model 2 (area = %0.4f)' % (roc_auc2))
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel('False Positive Rate',)
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", fontsize=12, frameon=False)

        plt.savefig(f'ROC_Curve_Period_{i}.png')
