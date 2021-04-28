.. _guidelines:

Guidelines
===========
There's a logic progression that should be used in order to obtain our results:

1. Run ``portfolioML.data.data_generator`` to obtain the csv file of price data.
2. Run ``portfolioML.data.data_returns``. It will generate all the csv files needed for the study.
3. At this point you have all you need to train one of the ML algorithms implemented.
   Go to one of their folder and run, for example, ``portfolioML.model.LSTM.lstm``.
   It will save all the trained models in .h5 format in a new folder named as you choosen
   and some plot of losses and accuracies in another one called 'accuracies_losses'.
4. Now that you have trained your kinky model you can go to the results folder and run
   ``portfolioML.results.model_predict``. It will return a csv file that contains the forecasts and use it to
   generate a ROC curve (Receiver operating characteristic curve). You have almost done.
5. Finally, the module ``portfolioML.results.portfolio_creation`` 



.. warning::
    The majority of the modules is strictly constrained on their path, since they pick and/or
    create folders and files starting from their position. Keep in mind to run them from their
    folder.

.. note::
   For every module, run at first

   .. code-block:: python

      python <module_name> -h

   to check the available options.
