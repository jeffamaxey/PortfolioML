## PortfolioML
[![Build Status](https://travis-ci.org/DanieleMDiNosse/PortfolioML.svg?branch=main)](https://travis-ci.org/DanieleMDiNosse/PortfolioML)
[![Documentation Status](https://readthedocs.org/projects/portfolioml/badge/?version=latest)](https://portfolioml.readthedocs.io/en/latest/?badge=latest)

## Table of contents
* [General info](#general-info)
* [Project structure](#project-structure)
* [Setup](#setup)

## General info
pass
	
## Setup
To run this project, clone the repository.
```
$ git clone https://github.com/DanieleMDiNosse/PortfolioML.git
```
If you use conda, you can create a new conda environment with:
```
$ conda create -n name_environment
$ conda activate name_environment
```
And then install with:
```
$ cd ../PortfolioML/
$ pip install .
```
## Project structure
```
├── docs
├── LICENSE
├── portfolioML
│   ├── data
│   │   ├── after_test.csv
│   │   ├── after_train.csv
│   │   ├── autoencoder_cnn.py
│   │   ├── autoencoders.py
│   │   ├── data_generator.py
│   │   ├── data_returns.py
│   │   ├── data_visualization.py
│   │   ├── MultidimReturnsData1.csv
│   │   ├── MultidimReturnsData2.csv
│   │   ├── MultidimReturnsData3.csv
│   │   ├── MultidimReturnsData4.csv
│   │   ├── preprocessing.py
│   │   ├── PriceData.csv
│   │   ├── ReturnsBinary.csv
│   │   ├── ReturnsBinaryPCA.csv
│   │   ├── ReturnsData.csv
│   │   └── ReturnsDataPCA.csv
│   ├── makedir.py
│   ├── model
│   │   ├── CNN
│   │   │   ├── CNN_dense
│   │   │   ├── CNN_dense+
│   │   │   ├── CNN_dense2_plus
│   │   │   ├── CNN_dense_pca_wave
│   │   │   ├── CNN_minpool
│   │   │   └── cnn.py
│   │   ├── DNN
│   │   │   ├── DNN_mymod2
│   │   │   ├── DNN_mymod2_pca
│   │   │   ├── DNN_mymod4
│   │   │   ├── DNN_paper
│   │   │   ├── DNN_paper_auto
│   │   │   ├── DNN_paper_pca
│   │   │   └── dnn.py
│   │   ├── LSTM
│   │   │   ├── LSTM_Model1
│   │   │   ├── LSTM_Model2
│   │   │   ├── LSTM_Model4
│   │   │   └── lstm.py
│   │   ├── preprocessing_ang.py
│   │   ├── RAF
│   │   │   └── raf.py
│   │   └── split.py
│   ├── results
│   │   ├── model_predict.py
│   │   ├── portfolio_creation.py
│   │   ├── predictions
│   │   │   ├── CNN
│   │   │   ├── DNN
│   │   │   ├── LSTM
│   │   │   └── RAF
│   │   ├── predictions_for_portfolio
│   │   │   ├── CNN
│   │   │   ├── DNN
│   │   │   ├── LSTM
│   │   │   └── RAF
│   │   ├── ROC
│   │   │   ├── CNN
│   │   │   ├── DNN
│   │   │   ├── LSTM
│   │   │   └── RAF
│   └── tests
├── README.md
├── requirements.txt
└── setup.py
```
