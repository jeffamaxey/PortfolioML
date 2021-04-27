.. PortfolioML documentation master file, created by
   sphinx-quickstart on Mon Apr 26 16:21:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PortfolioML's documentation!
=======================================
This package contains functions used to forecasts stock market returns on a fraction of the S&P500 index.
The procedure used is mainly focused on the following two articles:

Thomas Fischer, Christopher Krauss,
Deep learning with long short-term memory networks for financial market predictions,
European Journal of Operational Research,
Volume 270, Issue 2,
2018,
Pages 654-669,
ISSN 0377-2217,
https://doi.org/10.1016/j.ejor.2017.11.054.
(https://www.sciencedirect.com/science/article/pii/S0377221717310652)
Abstract: Long short-term memory (LSTM) networks are a state-of-the-art technique for sequence learning. They are less commonly applied to financial time series predictions, yet inherently suitable for this domain. We deploy LSTM networks for predicting out-of-sample directional movements for the constituent stocks of the S&P 500 from 1992 until 2015. With daily returns of 0.46 percent and a Sharpe ratio of 5.8 prior to transaction costs, we find LSTM networks to outperform memory-free classification methods, i.e., a random forest (RAF), a deep neural net (DNN), and a logistic regression classifier (LOG). The outperformance relative to the general market is very clear from 1992 to 2009, but as of 2010, excess returns seem to have been arbitraged away with LSTM profitability fluctuating around zero after transaction costs. We further unveil sources of profitability, thereby shedding light into the black box of artificial neural networks. Specifically, we find one common pattern among the stocks selected for trading – they exhibit high volatility and a short-term reversal return profile. Leveraging these findings, we are able to formalize a rules-based short-term reversal strategy that yields 0.23 percent prior to transaction costs. Further regression analysis unveils low exposure of the LSTM returns to common sources of systematic risk – also compared to the three benchmark models.
Keywords: Finance; Statistical arbitrage; LSTM; Machine learning; Deep learning

Christopher Krauss, Xuan Anh Do, Nicolas Huck,
Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500,
European Journal of Operational Research,
Volume 259, Issue 2,
2017,
Pages 689-702,
ISSN 0377-2217,
https://doi.org/10.1016/j.ejor.2016.10.031.
(https://www.sciencedirect.com/science/article/pii/S0377221716308657)


.. toctree::
   :maxdepth: 4
   :caption: Contents:

    source/modules
    source/portfolioML
    source/portfolioML.data
    source/portfolioML.model
    source/portfolioML.results

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
