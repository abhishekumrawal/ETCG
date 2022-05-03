# -*- coding: utf-8 -*-

import etcg_nk
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import pandas as pd
from matplotlib import ticker as mticker

def plot_lin_k():
    plt.rcParams['figure.figsize'] = [10,8]
    plt.rc('font', size = 30)
    plt.rc('legend', fontsize = 20)
    
    #===================================================================
    #Cumulative regret plot
    
    #seed
    s = random.randint(0,100000)
    
    cum_regret_etcg = []
    std_regret_etcg = []
    
    for K in [2,4,6,8,10]:  
        etcg_result = etcg_nk.cum_regret_multirun(20,100,K,100000,s)
        
        cum_regret_etcg.append(np.mean(np.sum(etcg_result, axis=1)))
    
        std_regret_etcg.append(np.std(np.sum(etcg_result, axis=1))/math.sqrt(20))
    
    mean_plot = pd.DataFrame(list(zip(cum_regret_etcg)),
                      columns =['etcg'], index=[2,4,6,8,10])
    
    mean_plot.index.name = 'horizon'
    
    mean_plot.to_csv('toy_lin_regret_mean_k.csv')
    
    err_plot = pd.DataFrame(list(zip(std_regret_etcg)),
                      columns =['etcg'], index=[2,4,6,8,10])
    
    err_plot.index.name = 'horizon'
    
    err_plot.to_csv('toy_lin_regret_err_k.csv')
    
    mean_plot = pd.read_csv('toy_lin_regret_mean_k.csv', index_col='horizon')
    
    err_plot = pd.read_csv('toy_lin_regret_err_k.csv', index_col='horizon')
    
    #estimate slop 
    y = [np.log(val) for val in list(mean_plot.etcg)]
    x = [np.log(val) for val in list(mean_plot.index)]
    m, b = np.polyfit(x, y, 1)
    
    x = np.linspace(2,10,100)
    y1 = 10*x**(4/3)
    y2 = 0.2*x**(4/3)
    y3 = 100*x**(4/3)
    y4 = 5*x**(4/3)
    y5 = 1000*x**(4/3)
    
    plot_T=['etcg']
    
    fig, ax = plt.subplots()
    for i in plot_T:
        #ax.plot(mean_plot[i],lw=5, marker='D',markersize=15)
        ax.errorbar(x=[2,4,6,8,10], y=mean_plot[i], yerr=err_plot[i],capsize=10,lw=5, marker='D',markersize=15,label=i)
        
    #ax.plot(range(1,100001), poly1d_fn(range(1,100001)), ls='--', lw='3', c='black', label="IMlinUCB")
    ax.set_xlabel('$k$')
    ax.set_ylabel('Cumulative Regret')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.grid(linestyle='-', linewidth=1)
    ax.set_title('Linear Reward')
    ax.legend(['ETCG'])
    
    #ax.plot(x,y1,lw=5,linestyle='dashed', alpha=0.5, color='grey')
    #ax.plot(x,y2,lw=3,linestyle='dashed', alpha=0.5, color='grey')
    ax.plot(x,y3,lw=5,linestyle='dashed', alpha=0.5, color='grey')
    #ax.plot(x,y4,lw=3,linestyle='dashed', alpha=0.5, color='grey')
    ax.plot(x,y5,lw=5,linestyle='dashed', alpha=0.5, color='grey')
    
    ax.get_figure().savefig('regret_lin_k.png', dpi=300,bbox_inches='tight')
