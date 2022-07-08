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
import utils as utl

import os

os.chdir('C:\\Users\\shube\\Documents\\GitHub\\CC\\')

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
        
def df_sig_sma_cr2(df_data, ma1, ma2, side):

    #compute sma x
    df_sma1 = df_sig_sma(df_data, ma1)
    df_sma2 = df_sig_sma(df_data, ma2)
    df_sma_cr = np.sign(df_sma1 - df_sma2)
    return df_sma_cr

if __name__ == '__main__':
    #set parameters
    DM_list_securities = ['SPY','TLT', 'TIP', 'GLD'] #20/120
    EM_list_securities = ['GXC','PCY', 'GLD'] #5/25
    list_securities = ['BKF','ILF', 'ASEA','AFK'] #5/25
    tkr = 'BKF'
    window = 30 #days

    #get market data
    data = df_yahoo_data(list_securities)
    data_returns = df_returns(data)  
    
    #setup inverse volatilty portfolio
    inv_volat_data = df_inverse_volatility(data_returns, window)
    data_norm_weights = df_weighted(inv_volat_data)
    data_iv_port_rets = df_port_returns(data_norm_weights, data_returns)
    
    #setup equally weighted portfolio
    data_eq_weight_rets = df_equal_weight_rets(data_returns)
    #data_iv_port_rets.dropna(inpl  

    #setup trend following portfolio
    ma_s = 5
    ma_l = 25
    sma_s = df_sig_sma(data, ma_s)
    sma_l = df_sig_sma(data, ma_l)
    sma_sig = df_sig_sma_cr(data, ma_s, ma_l)
    trend_port_rets = df_strat_returns(sma_sig, data_returns)
     
    if debug_xl == True:
        with pd.ExcelWriter('AW_dbg.xlsx',  date_format = 'dd-mm-yyyy', 
                            datetime_format='dd-mm-yyyy') as writer:
            data.to_excel(writer, sheet_name='prices')
            data_returns.to_excel(writer, sheet_name='returns')
            sma_s.to_excel(writer, sheet_name='sma_s')
            sma_l.to_excel(writer, sheet_name='sma_l')
            sma_sig.to_excel(writer, sheet_name='sma_cr_sig')
            trend_port_rets.to_excel(writer, sheet_name = 'strat_rets')      
            
            df_dbg = pd.DataFrame()     
            
            df_dbg['pr'] = data[tkr]
            df_dbg['r'] = data_returns[tkr]
            df_dbg['sma_s'] = sma_s[tkr]     
            df_dbg['sma_l'] = sma_l[tkr]
            df_dbg['sig'] = sma_sig[tkr]
            df_dbg['tot'] = trend_port_rets[tkr]
            df_dbg.to_excel(writer, sheet_name=tkr)  
 
    qs.reports.html(trend_port_rets['Tot'], benchmark = data_eq_weight_rets['Tot'] ,
                    output = 'no_none',  title='MATS performance',
                    download_filename='MATS_performance.html')
