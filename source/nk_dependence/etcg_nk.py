# -*- coding: utf-8 -*-

"Importing necessary modules"
import numpy as np
import random
import math
from random import seed
import heapq
from scipy.stats import truncnorm

def etcg(bandit, K, T):
    
    """
    ETC greedy
    bandit: a vector representing a bandit environment (noiseless)
    K: cardinality
    """
    
    #linear
    def reward(arms_list):
        r = 0
        for i in arms_list:
            r += bandit[i] + truncnorm.rvs(-0.1, 0.1, loc=0, scale=0.1)
        return(r/K)
    
    N = len(bandit)
    
    m = math.ceil(((T*math.sqrt(2*math.log(T)))/(N+2*N*K*math.sqrt(2*math.log(T))))**(2/3))

    selected_action_etcg = []
    obs_influences_etcg = []
    accept_set = []
    tbd_set = list(range(len(bandit)))
    time_step = 0 #time step counter
    
    for k in range(K):
        tbd_set_rewards = [0]*len(tbd_set)
        for i in range(len(tbd_set)):
            for count in range(m):
                reward_chosen_arms = reward(accept_set+[tbd_set[i]])
                chosen_arms = accept_set+[i]
                tbd_set_rewards[i] += reward_chosen_arms
                
                selected_action_etcg.append(chosen_arms)
                obs_influences_etcg.append(reward_chosen_arms)
                time_step += 1
                
            tbd_set_rewards[i] = tbd_set_rewards[i]/m
            
        new_acc_reward = np.max(tbd_set_rewards)
        new_accept_index = np.random.choice(np.where(tbd_set_rewards == new_acc_reward)[0])
        best_arm = tbd_set[new_accept_index]
        
        accept_set.append(best_arm)
        tbd_set.remove(best_arm)
        
    print(accept_set)
    opt_inf = 0
    
    #Remaining time        
    for t in range(time_step, T):
        selected_action_etcg.append(accept_set)
        infl = reward(accept_set)
        obs_influences_etcg.append(infl)
        opt_inf += infl
        
    print(opt_inf/(T-time_step))
        
    return selected_action_etcg, obs_influences_etcg

def cum_regret_multirun(n_runs, N, K, T, s):
    regret = []
    bandit = []
    
    seed(s)
    for i in range(N):
        bandit.append(random.uniform(0.1, 0.9))
    
    #linear
    opt = sum(heapq.nlargest(K, bandit))/K
    
    for _ in range(n_runs):  
        rewards = etcg(bandit, K, T)[1]
        regret.append([opt-x for x in rewards])
        
    a = np.array(regret)
    #result = np.average(a, axis=0)
    return a

