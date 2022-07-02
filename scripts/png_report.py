import quantstats as qs
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import seaborn as sns
import utils as utl
import html_evaluation_portfolio as hep
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from quantstats.stats import *
from quantstats._plotting.wrappers import distribution

def list_securities_name(list_securities):
    unique_name = ''
    for i in list_securities:
        unique_name += '_' + i
    return unique_name

def table_montly_returns(data_earnings, fig_size = (10, 5), fontname = 'Arial', annot_size = 10, square = False, cbar = False, save_fig = False, path = 'figure.png',
                        cmap = LinearSegmentedColormap.from_list('RedGreen', ['crimson', 'gold', 'lime']), eyo = False, compunded = True):

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title('Monthly Returns (%)\n', fontsize=14, y=.995,
                 fontname=fontname, fontweight='bold', color='black')

    returns = monthly_returns(data_earnings['Tot'], eoy=eoy,compounded=compounded) * 100

    #print(returns.div(100).add(1).cumprod(axis = 1)) #replace(0,1).cumprod(axis=1)['DEC']
    returns['YEAR'] = returns.div(100).add(1).cumprod(axis = 1)['DEC']
    returns['YEAR'] = returns['YEAR'].sub(1).mul(100)

    reversed_df = returns.iloc[::-1]

    #display(returns)

    ax = sns.heatmap(reversed_df, ax=ax, annot=True, center=0,
                    annot_kws={"size": annot_size}, vmax = 10,
                    fmt="0.2f", linewidths=2,
                    square=square, cbar=cbar, cmap = 'gray',
                    cbar_kws={'format': '%.0f%%'})

    ax.tick_params(colors="#808080")
    plt.xticks(rotation=0, fontsize=annot_size*1.2)
    plt.yticks(rotation=0, fontsize=annot_size*1.2)
    plt.subplots_adjust(hspace=0, bottom=0, top=1)
    fig.tight_layout(w_pad=0, h_pad=0)
    if save_fig:
        # path = './img/temp/table_montly_returns' + unique_name + '.png'
        plt.savefig(path)


def hist_returns(data_earnings, eoy=False, compounded=True, save_fig = False, path = 'figure.png', fill = False):
    returns = monthly_returns(data_earnings['Tot'], eoy=eoy,compounded=compounded) * 100

    series = pd.Series(dtype = 'float64')
    for column in returns.columns:
        series = pd.concat([series,returns[column]])
        #distribution(returns)
        ax = series.plot.hist(bins = 30, edgecolor='k',fill=fill)
        ax.set_title('Distribution of returns', size = 25)
        if fill:
            ax.axvline(0, color='w', linestyle='--')
        else:
            ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel("Returns in %")
        ax.set_ylabel("Frequency")
        fig = ax.get_figure()
        fig.set_size_inches(5.5,6)
        if save_fig:
            # path = './img/temp/distribution_returns' + unique_name + '.png'
            fig.savefig(path)


def table_stats(data, save_fig = False, path = 'figure.png'):
    statistics = pd.DataFrame(columns = ['index', 'value'])

    statistics.loc[0] = ['Annualised return', utl.annualized_return(data_earnings['Tot'])]
    statistics.loc[1] = ['Annualised volatility', utl.annualised_volatility(data_earnings['Tot'])]
    statistics.loc[2] = ['Sharpe ratio', sharpe(data_earnings['Tot'])]
    statistics.loc[3] = ['Sortino ratio', sortino(data_earnings['Tot'])]
    statistics.loc[4] = ['Adjusted sortino', adjusted_sortino(data_earnings['Tot'])]
    statistics.loc[5] = ['Skew', skew(data_earnings['Tot'])]
    statistics.loc[6] = ['Kurtosis', data_earnings['Tot'].kurt()]
    statistics.loc[7] = ['Max drawdown', max_drawdown(data_earnings['Tot'])]
    statistics.loc[8] = ['GPR', gain_to_pain_ratio(data_earnings['Tot'])]
    statistics.loc[9] = ['Calmar ratio', calmar(data_earnings['Tot'])]
    statistics.loc[10] = ['Pay-off ratio', payoff_ratio(data_earnings['Tot'])]

    if save_fig:
        #export as .csv
        #path = './data/temp/statistics_all_weather' + unique_name + '.csv'
        statistics.to_csv(path)

fig_underwater = drawdown(data_earnings['Tot'], grayscale = True, figsize=(10, 5.5), savefig =  './img/temp/under_water_plot' + unique_name + '.png')


def create_report_png(save_fig = False, path = 'report.png'):
    fig = plt.figure(figsize=(8, 13.55))

    gs = gridspec.GridSpec(3, 2, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[0, :])
    fig_table = plt.imread('./img/temp/table_montly_returns' + unique_name + '.png')
    ax1.imshow(fig_table)
    ax1.set_aspect('equal')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.grid(False)

    ax2 = plt.subplot(gs[1,:1])
    fig_distribiution = plt.imread('./img/temp/distribution_returns' + unique_name + '.png')
    ax2.imshow(fig_distribiution)
    ax2.set_aspect('equal')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid(False)

    ax3 = plt.subplot(gs[1, 1:])
    fig_stats = plt.imread('./img/temp/statistics_all_weather' + unique_name + '.png')
    ax3.imshow(fig_stats)
    ax3.set_aspect('equal')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.grid(False)

    ax4 = plt.subplot(gs[-1,:])
    fig_underwater = plt.imread('./img/temp/under_water_plot' + unique_name + '.png')
    ax4.imshow(fig_underwater)
    ax4.set_aspect('equal')
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.grid(False)

    if save_fig:
        #path = 'report' + unique_name + '.png'
        plt.savefig(path)
