""" Wavelet decomposition """
import argparse
import logging
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from portfolioML.makedir import smart_makedir, go_up

def approx_details_scale(data, wavelet, dec_level):
    """
    Approximation and details signal of a time series at specific time scale.

    Parameters
    ----------
    data: list, numpy array
        Input time-series data

    wavelet: string
        Wavelet's name used to decomposition. For all avelable wavelet see pywt.waveletlist().

    dec_level: integer
        level of time_scale on wich compute the approssimation and details analysis.
        Its range is [1, pywt.dwtn_max_level(data, wavelet)+1]

    Result
    ------
    approx: numpy arrey
        Approsimation values

    detail: numpy arrey
        Deatils values
    """ 

    max_level = pywt.dwtn_max_level(data, wavelet)
    logging.info(f'max_level:{max_level}')

    try:
        if dec_level > max_level + 1: raise ValueError
    except :
        print('dec_level is out of bound [1, max_level]')
        dec_level = max_level + 1 

    coeffs = pywt.wavedec(data, wavelet, level=dec_level)

    for i in range(2,len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])

    det = coeffs[1]

    coeffs[1] = np.zeros_like(coeffs[1])  
    approx = pywt.waverec(coeffs, wavelet)

    coeffs[1] = det
    coeffs[0] = np.zeros_like(coeffs[0])
    details = pywt.waverec(coeffs, wavelet)

    return approx, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This code process same price time-series data')
    parser.add_argument("wavelet", type=str, help="Type of wavelet")
    parser.add_argument("company", type=str, help="Selected company")
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
    df_price = pd.read_csv(go_up(1) + "/data/PriceData.csv")
    df_price = df_price.dropna(axis=1)    
    
    open_data = np.array(df_price[args.company])

    wave = args.wavelet
    max_level = pywt.dwtn_max_level(open_data, wave)
    coeffs = pywt.wavedec(open_data, wave, level=max_level)
    print(f'app + details:{len(coeffs)}')
    open_data_rec = pywt.waverec(coeffs, wave) # complete signal

    coeffs_1 = pywt.wavedec(open_data, wave, level=2)
    coeffs_1[2] = np.zeros_like(coeffs_1[2])
    coeffs_1[0] = np.zeros_like(coeffs_1[0])
    open_data_rec_1 = pywt.waverec(coeffs_1, wave)

    plt.figure('Total')
    plt.subplot(5,1,1)
    plt.plot(open_data, '-b', lw=0.8, label='real data')
    plt.plot(open_data_rec, '--r', lw=0.8, label='recostructed data')
    plt.title('Real data and Recostructed data')
    plt.legend()
    for scale in range(1, max_level+1):
        approx, details = approx_details_scale(open_data, wave, scale)
    
        plt.figure(f'{wave}-dec on scale {scale}')
        plt.subplot(2,1,1)
        plt.plot(approx, lw=0.8, label=f'approx d{scale}')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(details,lw=0.8, label=f'details d{scale}')
        plt.legend()
    
        plt.figure('Total')
        plt.subplot(5,1,scale+1)
        plt.plot(details, '-b', lw=0.8, label=f'det scale {scale}')
        plt.legend()


    plt.figure()
    plt.plot(open_data, '-b', lw=0.8, label='real data')
    approx, details = approx_details_scale(open_data, wave, 2)
    plt.plot(approx, lw=0.8, label=f'approx d2')
    plt.plot(details,lw=0.8, label='details d2')
    dif = np.array(open_data) - np.array(approx[:-1])
    plt.plot(dif, lw=0.8, label='dif')
    
    plt.legend()
    plt.show()