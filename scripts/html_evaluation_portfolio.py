import quantstats as qs
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from datetime import date
from IPython.display import display
from bgdataloader import polygon_fetch as pf


def create_dataframe(list_securities, start_date = pd.to_datetime('2007-01-03'), end_date = date.today()):
    df = pd.DataFrame()
    for security in list_securities:
        SEC = yf.Ticker(security)
        if start_date:
            hist = SEC.history(start = start_date, end = end_date)
        else:    
            hist = SEC.history(period="max")
        df[security] = hist['Close'] #I just consider the close price, is that correct?
        #df.dropna() #if this line the data are only from 2010 otyherwise some columns arrive until 2006
    return df

def create_dataframe_polygon(list_securities, start_date = '2007-01-03', end_date = str(date.today())):
    df = pd.DataFrame()
    for security in list_securities:
        pc = pf.PolygonClient()
        client = pc.get_client()
        hist = pc.get_equity_bar_data(security, 1, "day",start_date, end_date)
        
        df[security] = hist['close'] #I just consider the close price, is that correct?
        #df.dropna() #if this line the data are only from 2010 otyherwise some columns arrive until 2006
    return df

def return_df(df):
    return df.div(df.shift(1))-1


def df_inverse_volatility(df_perc, window):

    #compute the  1/ volatility (standard_deviation) over a range of window days
    return df_perc.rolling(window).apply(lambda x: 1/(x.std()))


def df_wheighted(inv_volat_df):

    #create a new df with the same columns
    df = pd.DataFrame(columns = inv_volat_df.columns)
    inv_volat_df.fillna(0, inplace = True)
    #itarate over row (always avoid it, if possible)
    for index, row in inv_volat_df.iterrows():

        #append the new value at each row
        df.loc[index] = row.div(sum(list(row)))
    return df


def df_earnings(df_weight_portf, data_perc):

    #compte the earning dataframe (df_weight*df_returns - entries by entries)
    #shift by 1 entries since I rebalance the portfolio the day after
    df = df_weight_portf.shift(1)*data_perc
    df['Tot'] = df.sum(axis= 1)
    return df


def df_earnings_equal_weight(df_percentage):

    #compute a dataframe with all equal weight for benchmarking
    df = df_percentage.mul(1/len(df_percentage.columns))
    df['Tot'] = df.sum(axis= 1)
    return df

def transaction_cost_computation(df_weighted, rebalance, initial_capital):
    # df where store the result
    quantity_to_balance = pd.DataFrame(columns = df_weighted.columns)
    
    df_weighted.dropna(inplace = True)
    df_weighted = df_weighted[start_date: end_date]
    perc_ret = data_earnings['Tot'].add(1).cumprod()
    
    #compute the capital -it changes during the investing time-
    capital =  perc_ret*initial_capital
    
    # cycle every time the portfolio is rebalanced
    for index in range(len(df_weighted)//rebalance):
        transition_array = []
        #cycle over the securities
        for sec in df_weighted.columns: 
            if index > 0: 
                # append the difference between the value before and after the rebalance
                transition_array.append(df_weighted[sec].iloc[index*rebalance+1]-df_weighted[sec].iloc[index*rebalance-1])
            if len(transition_array) == len(quantity_to_balance.columns):
                df_to_append = pd.DataFrame([transition_array], columns = df_weighted.columns, index = [df_weighted.index.tolist()[index*rebalance]])
                # append in the final dataframe
                quantity_to_balance = pd.concat([quantity_to_balance, df_to_append])
                # quantity to balance: percetange of capital to trade for balancing the portfolio
    
    
    capital = capital.loc[quantity_to_balance.index.to_list()]
    quantity_to_balance = quantity_to_balance.mul(capital, axis = 0)
    
    number_of_sec_exchange = pd.DataFrame()
    
    # iterate over the index and the row 
    for index, row in quantity_to_balance.iterrows():
        number_of_sec_exchange_array = []
        # repete the passages for each security
        for sec in quantity_to_balance.columns:
            # append to the array the money I have to move over the price that day 
            # To get the numbers of trades necessary that day for the specific security
            number_of_sec_exchange_array.append(row[sec]/data.loc[index][sec])
        if len(number_of_sec_exchange_array) == len(quantity_to_balance.columns): # when it is done for every security, append the result to the dataframe
            #create dataframe to concatenate
            df_to_append = pd.DataFrame([number_of_sec_exchange_array], columns = quantity_to_balance.columns, index = [index])
            # concatenate the existence dataframe with the enw one
            number_of_sec_exchange = pd.concat([number_of_sec_exchange, df_to_append])    
            # number_of_sec_exchange: number of security necessary to exchange to rebalance the porfolio
    return number_of_sec_exchange


if __name__ == '__main__':
    #set parameters
    list_securities = ['SSO','UBT', 'UST', 'UGL']
    window = 30 #days

    #main
    data = create_dataframe(list_securities)
    data_perc = return_df(data)
    inv_volat_data = df_inverse_volatility(data_perc, window)
    data_wheighted = df_wheighted(inv_volat_data)
    data_earnings = df_earnings(data_wheighted, data_perc)
    data_earn_eq_weight = df_earnings_equal_weight(data_perc)
    data_earnings.dropna(inplace = True)
    data_earn_eq_weight.dropna(inplace = True)
    print(data_earn_eq_weight)
    qs.reports.html(data_earnings['Tot'], benchmark = data_earn_eq_weight['Tot'] ,
                    output = 'no_none',  title='All weather performance',
                    download_filename='all_weather_performance.html')
