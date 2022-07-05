import quantstats as qs
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from datetime import date
from IPython.display import display
#from bgdataloader import polygon_fetch as pf
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.chdir('C:\\Users\\shube\\Documents\\GitHub\\CCS\\')

debug_xl = True


def df_yahoo_data(list_securities, start_date = pd.to_datetime('2005-01-01'), end_date = date.today()):
    df = pd.DataFrame()
    for security in list_securities:
        SEC = yf.Ticker(security)
        if start_date:
            hist = SEC.history(start = start_date, end = end_date)
        else:    
            hist = SEC.history(period="max")
        df[security] = hist['Close'] #I just consider the close price, is that correct?
        #df.dropna() #if this line the data are only from 2010 otyherwise some columns arrive until 2006
    df = df.set_index(pd.to_datetime(df.index,format="%Y-%m-%d"))
   # [dt.to_datetime().date() for dt in df.dates]
    return df

def df_polygon_data(list_securities, start_date = '2007-01-03', end_date = str(date.today())):
    df = pd.DataFrame()
    for security in list_securities:
        pc = pf.PolygonClient()
        client = pc.get_client()
        hist = pc.get_equity_bar_data(security, 1, "day",start_date, end_date)
        
        df[security] = hist['close'] #I just consider the close price, is that correct?
        #df.dropna() #if this line the data are only from 2010 otyherwise some columns arrive until 2006
    return df

def df_returns(df):
    return df.div(df.shift(1))-1


def df_inverse_volatility(df_perc, window):

    #compute the  1/ volatility (standard_deviation) over a range of window days
    return df_perc.rolling(window).apply(lambda x: 1/(x.std()))


def df_weighted(inv_volat_df):

    #create a new df with the same columns
    df = pd.DataFrame(columns = inv_volat_df.columns)

    #itarate over row (always avoid it, if possible)
    for index, row in inv_volat_df.iterrows():

        #append the new value at each row
        df.loc[index] = row.div(sum(list(row)))
    return df


def df_port_returns(df_weight_portf, data_perc):

    #compte the earning dataframe (df_weight*df_returns - entries by entries)
    #shift by 1 entries since I rebalance the portfolio the day after
    df = df_weight_portf.shift(1)*data_perc
    df['Tot'] = df.sum(axis= 1)
    return df


def df_strat_returns(df_positions, data_returns):

    df = df_positions.shift(1) * data_returns
    df['Tot'] = df.sum(axis= 1) / len(df.columns)
    df['cumR'] = df['Tot'].add(1).cumprod()
    return df


def df_equal_weight_rets(df_percentage):

    #compute a dataframe with all equal weight for benchmarking
    df = df_percentage.mul(1/len(df_percentage.columns))
    df['Tot'] = df.sum(axis= 1)
    return df


def df_sig_sma(df_data, ma):
    #compute sma with window ma
    df_sma = pd.DataFrame()
    for col in df_data:
        df_sma[col] = ta.SMA(df_data[col].to_numpy(), ma)   
    df_sma.index = df_data.index    
    return df_sma


def thresh_sig(x,z):
    if x > z:
        return 1
    elif x < z:
        return -1
    else:
        return 0


def df_sig_sma_cr(df_data, ma1, ma2):

    #compute sma x
    df_sma1 = df_sig_sma(df_data, ma1)
    df_sma2 = df_sig_sma(df_data, ma2)
    df_sma_cr = np.sign(df_sma1 - df_sma2)
    return df_sma_cr
        

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
    list_securities = ['SPY','TLT', 'TIP', 'GLD']
    window = 30 #days

    #main
    data = df_yahoo_data(list_securities)
    data_returns = df_returns(data)  
    
    inv_volat_data = df_inverse_volatility(data_returns, window)
    data_norm_weights = df_weighted(inv_volat_data)
    data_iv_port_rets = df_port_returns(data_norm_weights, data_returns)
    
    data_eq_weight_rets = df_equal_weight_rets(data_returns)
    #data_iv_port_rets.dropna(inpl  

    ma_s = 20
    ma_l = 120
    sma_s = df_sig_sma(data, ma_s)
    sma_l = df_sig_sma(data, ma_l)
    sma_sig = df_sig_sma_cr(data, ma_s, ma_l)
    
    strat_rets = df_strat_returns(sma_sig, data_returns)
     
    if debug_xl == True:
        with pd.ExcelWriter('AW_dbg.xlsx',  date_format = 'dd-mm-yyyy', 
                            datetime_format='dd-mm-yyyy') as writer:
            data.to_excel(writer, sheet_name='prices')
            data_returns.to_excel(writer, sheet_name='returns')
            sma_s.to_excel(writer, sheet_name='sma5')
            sma_l.to_excel(writer, sheet_name='sma25')
            sma_sig.to_excel(writer, sheet_name='sma_cr_sig')
            strat_rets.to_excel(writer, sheet_name = 'strat_rets')
          
    #data_eq_weight_rets.dropna(inplace = True)
   # print(data_eq_weight_rets)
        
        
    qs.reports.html(strat_rets['Tot'], benchmark = data_eq_weight_rets['Tot'] ,
                    output = 'no_none',  title='MATS performance',
                    download_filename='MATS_performance.html')
