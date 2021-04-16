import argparse
import logging
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from makedir import smart_makedir, go_up


def get_trading_values(df_price, algorithm, model_name, len_period=1308, len_train=981, len_test=327):
    '''
    Generate a pandas dataframe composed by all the days of which lstm.py forecasts
    the prices. This is due to the fact that lstm.py doesn't track the dates'

    Parameters
    ----------
    df_price : str
        Path of the csv file of prices.
    algorithm : str
        LSTM, DNN or CNN.
    model_name : str
        Name of the particular selected model trained. Check the name of folders in predictions.
    len_period : int, optional
        Lenght of each study period. The default is 1308.
    len_train : int, optional
        Lenght of the training set. The default is 981.
    len_test : int, optional
        Lenght of the trading set. The default is 327.

    Returns
    -------
    None.

    '''
    len_total_leave = len(df_price)-len_period

    # Divide in periods the price dataframe
    periods_price = [(df_price[i:len_period+i]) for i in range(0, len_total_leave+1, len_test)]

    # Select only the test sets
    tests = [periods_price[i][len_train:] for i in range(len(periods_price))]

    # Select, then, only days in test sets of which forecasts are actually made
    trading_values = [tests[i][240:] for i in range(len(tests))]

    smart_makedir(f'predictions_for_portfolio/{algorithm}/{model_name}')
    path1 = os.getcwd() + f'/predictions_for_portfolio/{algorithm}/{model_name}'

    # Insert the 'Date' column in the forecasts made by lstm.py
    for i in range(10):
        ith_predictions = pd.read_csv(f"predictions/{algorithm}/{model_name}/{model_name}_Predictions_{i}th_Period.csv",
                                      index_col=0)
        ith_predictions.insert(0,'Date',trading_values[i]['Date'].values)

        # Save the csv file
        ith_predictions.to_csv(f"{path1}/Trading_days_period{i}.csv")
    logging.info("Successfully tracked forecasts dates")

def portfolio_creation(algorithm, model_name, num_periods, k=10):
    '''
    This function creates a list composed by pandas dataframe each of which contains
    the top k and bottom k forecasts made by the LSTM algorithm in lstm.py in the whole
    trading period.

    Parameters
    ----------
    forecasts : str
        Path to the predictions csv file.
    k : int, optional
        Number of top and flop forecasts that will be considered. The default is 10.

    Returns
    -------
    portfolio : list
        List of pandas dataframes.
    '''

    get_trading_values(df_price, algorithm, model_name)
    path = os.getcwd() + f'/predictions_for_portfolio/{algorithm}/{model_name}'
    portfolio = []
    for j in range(num_periods):
        trading_days = pd.read_csv(f"{path}/Trading_days_period{j}.csv", index_col=0)
        portfolio_tmp = []
        for i in range(trading_days.shape[0]):
            df_portfolio = pd.DataFrame()

            # Select the ith day
            day = trading_days['Date'][i]

            # Select and order the values of that day
            values = trading_days.iloc[i][1:]
            values = values.sort_values()

            # Select top e bottom k values and corresponding companies
            top_k_val, top_k_comp = values[:k], values[:k].index
            bottom_k_val, bottom_k_comp = values[-k:], values[-k:].index
            values_traded = list(top_k_val.values) + list(bottom_k_val.values)
            companies_traded = list(top_k_comp) + list(bottom_k_comp)

            # Put all them in a dataframe
            df_portfolio[day] = values_traded
            df_portfolio['Company'] = companies_traded
            portfolio_tmp.append(df_portfolio)
        portfolio.append(portfolio_tmp)
    return portfolio

def forecast_returns(df_price, num_periods=10, k=10, money=1., monkey=False):
    '''
    The following is aimed to calculate the daily returns. We set a long position for the
    top k companies at each day and a short position for the bottom k ones. So, we
    calculate the return using prices ad t ad t+1 for the formers and t-1 ad t for the
    latters

    Parameters
    ----------
    num_periods : int, optional
        Number of period over which forecast returns have to be calculated. The default is 10
    k : int,  optional
        Number of top and flop forecasts that will be considered. The default is 10.
    money : float, optional
        How much you want to invest

    Returns
    -------
    returns : numpy array
        Numpy array of all returns for all periods
    accumulative_returns : numpy array
        Accumulative returns
    '''
    global num_periods_g
    global k_g
    global money_g

    num_periods_g = num_periods
    k_g = k
    money_g = money

    returns = []
    for period in range(num_periods):
        for i in range(len(portfolio[0])):

            # Select one trading day
            trading_day = portfolio[period][i].columns[0]
            if monkey:
                rand = random.sample(range(1,len(df_price.columns)),2*k)
                companies = df_price.columns[rand]
            else:
                # List all companies in that day
                companies = portfolio[period][i].Company.tolist()

                # Select the corresponding index in df_price
            index = df_price.Date[df_price.Date == trading_day].index.tolist()
            index = index[0]

            # Determine the returns for long and short positions
            for i,comp in enumerate(companies):
                if i <=9:
                    returns.append(df_price[comp][index]/df_price[comp][index+1] - 1)
                else:
                    returns.append(df_price[comp][index+1]/df_price[comp][index] - 1)
    returns = np.array(returns)
    returns_rs = np.reshape(returns, (int(len(returns)/(2*k)),(2*k)))

    #Accumulative returns
    accumulative_returns = []
    for day_returns in returns_rs:
        money = money + ((money/(2*k))*day_returns).sum()
        accumulative_returns.append(money)
    accumulative_returns = np.array(accumulative_returns)

    #Mean daily returns
    returns_dr = np.reshape(returns, (int(returns.shape[0]/(2*k)),(2*k)))
    mean_daily_returns = [day_ret.mean() for day_ret in returns_dr]

    logging.info('Average daily returns %2f', returns.mean())
    logging.info('Standard deviation %2f', returns.std())

    return returns, accumulative_returns

def statistical_significance(df_price, monkeys_num, num_periods, k):
    """
    Compute statistical significance of selected model.

    Compare the average daily return of model with average daily return of several
    trading monkey and compute the Z-score test for statistical significance.
    If p value is grather than 0.05 so the model basically is equal to a monkey.

    Parameters
    ----------
    monkeys_nun: integer
        How many monkeys do you want?

    Results
    -------

    """


    # Monkey statistic
    returns_dr = []
    for i in range(0,monkeys_num):
        returns, acc_returns_m = forecast_returns(df_price, num_periods=num_periods_g,
                                                        money=money_g, monkey=True)
        returns = np.reshape(returns, (int(returns.shape[0]/(2*k_g)),(2*k_g)))
        returns_dr.append(returns)
    returns_dr = np.array(returns_dr)
    returns_dr = np.reshape(returns_dr, (returns_dr.shape[0]*returns_dr.shape[1], returns_dr.shape[2]))
    mean_daily_ret = np.array([day_ret.mean() for day_ret in returns_dr])


    # Model statistic
    returns_mod, accumulative_returns_mod = forecast_returns(df_price, num_periods=num_periods_g,
                                                               money=money_g, monkey=False)
    returns_mod = np.reshape(returns_mod, (int(returns_mod.shape[0]/(2*k_g)),(2*k_g)))
    mean_return_mod = np.array([day_ret.mean() for day_ret in returns_mod])

    # Z-score test
    t_stat, p_val = stats.ttest_ind(mean_daily_ret, mean_return_mod, equal_var=False)

    return mean_daily_ret, mean_return_mod, p_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creation of portfolios based on selected model predictions and plot basic statistical')
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('--algorithm', '-a', type=str, action='append', help='CNN, LSTM , DNN and/or RAF')
    requiredNamed.add_argument('--model_name', '-m', type=str, action='append', help='Select the particular model trained')
    parser.add_argument("--num_periods", '-p', type=int, default=10, help="Number of period over which returns have to be calculated ")
    parser.add_argument("--money", '-€', type=int, default=1, help="How much you want to invest")
    parser.add_argument("--top_bottom", '-tp', type=int, default=10, help="Number of top (long pos) and bottom (short pos)")
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("--monkeys_num", type=int, default=10, help="How many monkeys do you want?")
    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])

    df_price = pd.read_csv(go_up(1) + "/data/PriceData.csv")
    df_price = df_price.dropna(axis=1)

    i = 0
    for alg, mod in zip(args.algorithm, args.model_name):
        i += 1
        logging.info(f"---------- Model {mod} ----------")
        path = os.getcwd() + f'/predictions_for_portfolio/{alg}/{mod}'

        # Portfolios Generator
        portfolio = portfolio_creation(alg, mod, args.num_periods)

        # Accumulate Returns
        # returns_monkey, acc_returns_monkey = forecast_returns(df_price, num_periods=args.num_periods,
                                                                    # money=args.money, monkey=args.monkey)
        #List of trading day:
        flat_list = [item for sublist in portfolio for item in sublist]
        trading_days = [item.columns[0] for item in flat_list]
        x_label_position = np.arange(0,len(trading_days),50)
        x_label_day = [trading_days[i] for i in x_label_position]

        # Accumulate Returns
        plt.figure("Accumulative Returns", figsize=[13.,10.])
        if alg == args.algorithm[0]:
            acc_list = []
            for i in range(args.monkeys_num):
                returns_monkey, acc_returns_monkey = forecast_returns(df_price, num_periods=args.num_periods,
                                                                        money=args.money, monkey=True)
                acc_list.append(np.array(acc_returns_monkey))
            acc_list = np.array(acc_list)

            acc_monkey_mean = np.mean(acc_list, axis=0)

            acc_monkey_std = np.std(acc_list, axis=0)
            monkey_std_upper = (acc_monkey_mean + acc_monkey_std)
            monkey_std_lower = (acc_monkey_mean - acc_monkey_std)
            plt.plot(acc_monkey_mean, color='crimson', label='Monkeys')
            plt.fill_between(list(range(0,len(acc_monkey_mean))),monkey_std_upper, monkey_std_lower, color='crimson', alpha=0.2,
                        label=r'$\pm$ 1 std. dev.')

        returns_model, acc_returns_model = forecast_returns(df_price, num_periods=args.num_periods, money=args.money, monkey=False)


        plt.plot(acc_returns_model, label=f'{mod}')
        plt.title("Accumulative Returns over Trading Days")
        plt.xticks(x_label_position, x_label_day, rotation=60)
        plt.xlabel("Trading days")
        plt.ylabel("Accumulative Returns")
        plt.grid(True)
        plt.legend()
        plt.savefig("Accumulative returns")

        # Statistic
        a, b, p_val= statistical_significance(df_price, monkeys_num=args.monkeys_num, num_periods=args.num_periods, k=args.top_bottom)

        fig, ax1 = plt.subplots(figsize=[8.0, 6.0])
        ax1.hist(a, bins=150, color='crimson',label=f'Monkey return: {a.mean():.5f} $\pm${a.std():.5f}', alpha = 0.9)
        ax2 = ax1.twinx()
        ax2.hist(b, bins=70, color='green', label=f'{mod} Return: {b.mean():.5f} $\pm${b.std():.5f}', alpha=0.5)
        plt.title(f'{mod} significant statistic w.r.t {args.monkeys_num} monkeys')
        ax1.plot([], color='white', label=f'p-value: {p_val:.4E}')
        ax1.set(xlabel='Average daily return')
        ax1.set(ylabel='Monkeys')
        ax2.set(ylabel='Model')
        fig.legend(bbox_to_anchor=(1.0,1.0), bbox_transform=ax1.transAxes)
        fig.savefig(f"Statistics model: {mod}")
    plt.show()
