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
│   │   ├── preprocessing.py
│   │   ├── autoencoder_cnn.py
│   │   ├── autoencoders.py
│   │   ├── data_generator.py
│   │   ├── data_returns.py
│   │   ├── after_test.csv
│   │   ├── MultidimReturnsData1.csv
│   │   ├── MultidimReturnsData2.csv
│   │   ├── MultidimReturnsData3.csv
│   │   ├── MultidimReturnsData4.csv
│   │   ├── PriceData.csv
│   │   ├── ReturnsBinary.csv
│   │   ├── ReturnsBinaryPCA.csv
│   │   ├── ReturnsData.csv
│   │   └── ReturnsDataPCA.csv
│   ├── makedir.py
│   ├── model
│   │   ├── CNN
│   │   ├── DNN
│   │   ├── LSTM
│   │   ├── RAF
│   │   └── split.py
│   ├── results
│   │   ├── model_predict.py
│   │   ├── portfolio_creation.py
│   │   ├── predictions
│   │   ├── predictions_for_portfolio
│   │   ├── ROC
│   └── tests
│       ├── test_data_returns.py
│       └── test_split.py
├── README.md
├── requirements.txt
└── setup.py
```

