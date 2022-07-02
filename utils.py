import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def max_dd(returns):
    """
    Compute max drawdown information for a time series
    
    Parameters
    ----------
    Input: returns = Series
    Output: scalar
    -------
    """
    
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.idxmin()
    start = r.loc[:end].idxmax()
    last = dd.tail(1).item()
    return mdd, start, end, last
    
    
def max_dd_sc(returns):
    """
    Compute max drawdown information for a time series
    
    Parameters
    ----------
    Input: returns = Series
    Output: scalar
    -------
    """
    
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    return mdd
   
    
def max_dd_ts(returns):
    """
    Compute drawdown graphs
    
    Parameters
    ----------
    Input: returns = Series
    Output: Series
    -------
    """
    
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    return dd


def max_dd_df(returns): # requires update
    """
    Compute max drawdowns on columns of a df
    
    Parameters
    ----------
    Input: returns = Dataframe
    Output: Dataframe
    -------
    """
    
    series = lambda x: pd.Series(x, ['Draw Down'])
    return returns.apply(max_dd).apply(series)


def freq_of_ts(se):
    """
    Compute no. of pionts per calendar year
    
    Parameters
    ----------
    Input: se = Series
    Output: scalar (no. of points per cal year)
    -------
    """
    
    start, end = se.index.min(), se.index.max()
    diff = end - start
    return round((len(se)-1) * 365.25 / diff.days, 2)


def annualized_return(df):
    """
    Compute annualised returns
    
    Parameters
    ----------
    Input: df = Dataframe
    Output: scalar 
    -------
     """
    
    freq = freq_of_ts(df)
    return df.mean() * freq
 

def annualised_volatility(df):
    """
    Compute annualised volatility
    
    Parameters
    ----------
    Input: df = Dataframe
    Output: scalar
    -------
    """
    
    freq = freq_of_ts(df)
    return df.std() * (freq ** .5)
    

def sharpe_ratio(df):
    """
    Compute annualised Sharpe Ratio
    
    Parameters
    ----------
    Input: df = Dataframe
    Output: Sharpe Ratio
    -------

    """    
    
    return annualized_return(df)/ annualised_volatility(df)


def describe_sr(df):
    """
    Compute performance statistics
    
    Parameters
    ----------
    Input: df = Dataframe
    Output: Dataframe
    -------

    """

    r = annualized_return(df).rename('Return')
    v = annualised_volatility(df).rename('Volatility')
    s = sharpe_ratio(df).rename('Sharpe')
    skew = df.skew().rename('Skew')
    kurt = df.kurt().rename('Kurt')
    dd = df.apply(max_dd_sc).rename('Max DD')
    desc = df.describe().T
    return pd.concat([ r, v, s, skew, kurt, dd, desc], axis=1).T.drop('count')

