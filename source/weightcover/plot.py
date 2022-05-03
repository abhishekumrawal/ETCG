# -*- coding: utf-8 -*-

import etcg
import dart
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

def plot_weightcover():
    plt.rcParams['figure.figsize'] = [10,8]
    plt.rc('font', size = 30)
    plt.rc('legend', fontsize = 20)
    
    cum_regret_etcg = []
    cum_regret_dart = []
    
    std_regret_etcg = []
    std_regret_dart = []
    
    for T in [100, 1000, 10000, 100000, 1000000]:  
        etcg_result = etcg.cum_regret_multirun(20,4,T)
        dart_result = dart.cum_regret_multirun(20,4,T)
        
        cum_regret_etcg.append(np.mean(np.sum(etcg_result, axis=1)))
        cum_regret_dart.append(np.mean(np.sum(dart_result, axis=1)))
    
        std_regret_etcg.append(np.std(np.sum(etcg_result, axis=1))/math.sqrt(20))
        std_regret_dart.append(np.std(np.sum(dart_result, axis=1))/math.sqrt(20))
    
    colnames =['etcg', 'dart']
    mean_plot = pd.DataFrame(list(zip(cum_regret_etcg, cum_regret_dart)),
                      columns = colnames, index=[100, 1000, 10000, 100000, 1000000])
    mean_plot.index.name = 'horizon'
    mean_plot.to_csv('toy_lin_regret_mean.csv')
    #mean_plot.reset_index(inplace=True)
    
    err_plot = pd.DataFrame(list(zip(std_regret_etcg, std_regret_dart)),
                      columns =colnames, index=[100, 1000, 10000, 100000, 1000000])
    err_plot.index.name = 'horizon'
    err_plot.to_csv('toy_lin_regret_err.csv')
    err_plot.reset_index(inplace=True)
    
    mean_plot = pd.read_csv('toy_lin_regret_mean.csv', index_col='horizon')
    
    #mean plot slope calculation
#    slopes = dict()
#    for algo in colnames:
#        y = [np.log(val) for val in list(mean_plot[algo])[3:]]
#        x = [np.log(val) for val in list(mean_plot.index)[3:]]
#        m, b = np.polyfit(x, y, 1)
#        slopes[algo] = round(m,4)
#    print(slopes)
#    with open('slopes.csv', 'w') as f:
#        for key in slopes.keys():
#            f.write("%s,%s\n"%(key,slopes[key]))
    
    err_plot = pd.read_csv('toy_lin_regret_err.csv', index_col='horizon')
    
    x = np.linspace(100,1000000,1000000)
    y1 = 0.1*x**(2/3)
    y2 = 0.2*x**(2/3)
    y3 = x**(2/3)
    y4 = 5*x**(2/3)
    y5 = 10*x**(2/3)
    
    plot_T=['etcg','dart']
    
    fontsize = 22
    fig, ax = plt.subplots()
    for i in plot_T:
        #ax.plot(mean_plot[i],lw=5, marker='D',markersize=15)
        ax.errorbar(x=[100, 1000, 10000, 100000, 1000000], y=mean_plot[i], yerr=err_plot[i],capsize=10,lw=5, marker='D',markersize=15,label=i)
        
    #ax.plot(range(1,100001), poly1d_fn(range(1,100001)), ls='--', lw='3', c='black', label="IMlinUCB")
    ax.set_xlabel('$T$', fontsize=fontsize)
    ax.set_ylabel('Cumulative Regret', fontsize=fontsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(linestyle='-', linewidth=1)
    ax.set_title('Product Recommendation', fontsize=fontsize)
    ax.legend(['ETCG','DART'])
    
    ax.plot(x,y1,lw=5,linestyle='dashed', alpha=0.5, color='grey')
    #ax.plot(x,y2,lw=3,linestyle='dashed', alpha=0.5, color='grey')
    ax.plot(x,y3,lw=5,linestyle='dashed', alpha=0.5, color='grey')
    #ax.plot(x,y4,lw=3,linestyle='dashed', alpha=0.5, color='grey')
    ax.plot(x,y5,lw=5,linestyle='dashed', alpha=0.5, color='grey')
    
    ax.get_figure().savefig('toy_regret_lin.png', dpi=300,bbox_inches='tight')
    
    #============================================================
    #Instantaneous reward plot
    def ma(x,d=100):
            i = 0
            ma_x = []
            while i < len(x) - d + 1:
                ma_x.append(sum(x[i : i + d]) / d)
                i += 1
            return ma_x
        
    T = 100000
    
    etcg_result = []
    dart_result = []
    
    bandit = [val for val in range(20)]
    
    ### opt calculation
    # =============================================================================
    # K = 2
    # group_size = 3
    # def split(list_a, chunk_size):
    #       for i in range(0, len(list_a), chunk_size):
    #         yield list_a[i:i + chunk_size]
    # groups = list(split(bandit, group_size))
    # #values = [round(x,2) for x in np.linspace(0,.5,len(groups))]
    # values = [round(.1*(val+1),1) for val in range(len(groups))]
    # opt = np.mean(sorted(values,reverse=True)[:K])
    # =============================================================================
    opt = .25
    
    for _ in range(20):    
        etcg_result.append(etcg.etcg(bandit, 4, T)[1])
        dart_result.append(dart.dart(bandit, 4, T)[1])
    
    reward_etcg = ma(np.mean(etcg_result, axis=0))
    reward_dart = ma(np.mean(dart_result, axis=0))
    
    
    rewards = pd.DataFrame(list(zip(reward_etcg, reward_dart)),
                      columns =['etcg','dart'])
    
    rewards.to_csv('toy_lin_reward_mean.csv')
    
    std_etcg = ma(np.std(etcg_result, axis=0)/math.sqrt(20))
    std_dart = ma(np.std(dart_result, axis=0)/math.sqrt(20))
    
    err = pd.DataFrame(list(zip(std_etcg, std_dart)),
                      columns =['etcg', 'dart'])
    
    err.to_csv('toy_lin_reward_err.csv')
    
    rewards = pd.read_csv('toy_lin_reward_mean.csv')
    
    err = pd.read_csv('toy_lin_reward_err.csv')
    
    fig, ax = plt.subplots()
    ax.plot(rewards['etcg'],lw=5)
    ax.fill_between(x=range(99901), y1=rewards['etcg']-err['etcg'],y2=rewards['etcg']+err['etcg'],alpha=0.2)
    ax.plot(rewards['dart'],lw=5)
    ax.fill_between(x=range(99901), y1=rewards['dart']-err['dart'],y2=rewards['dart']+err['dart'],alpha=0.2)
    
    ax.axhline(y=opt,xmin=0,xmax=100000,ls='--', lw=5,color='grey')
    ax.set_xlabel('$t$', fontsize=fontsize)
    ax.set_ylabel('Instantaneous Reward', fontsize=fontsize)
    ax.grid(linestyle='-', linewidth=1)
    ax.set_title('Product Recommendation', fontsize=fontsize)
    ax.legend(['ETCG', 'DART'],loc='lower right')
    
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))  
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  
    ax.get_figure().savefig('reward_plot_toy_lin.png', dpi=300, bbox_inches='tight')
