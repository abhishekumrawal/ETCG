# -*- coding: utf-8 -*-

import numpy as np
import random
import math
#from random import seed
#import heapq

def etcg(bandit, K, T):
    
    """
    ETC greedy
    bandit: a vector representing a bandit environment (noiseless)
    K: cardinality
    """
    
    def reward(arms_list):
        
        group_size = 6
        def split(list_a, chunk_size):
              for i in range(0, len(list_a), chunk_size):
                yield list_a[i:i + chunk_size]
        groups = list(split(bandit, group_size))
        
        #values = [round(x,2) for x in np.linspace(0,.5,len(groups))]
        values = [round(.1*(val+1),1) for val in range(len(groups))]
        
        group_ids_arms_list = []
        for i in arms_list:
            for j,group in enumerate(groups):
                if i in group: 
                    group_ids_arms_list.append(j)
        r = 0
        for i in list(set(group_ids_arms_list)):
            r += values[i] + random.uniform(-values[i], values[i])
        
        return r/K
    
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

def cum_regret_multirun(n_runs, K, T):
    regret = []
    bandit = [val for val in range(20)]
# =============================================================================
#     group_size = 6
#     def split(list_a, chunk_size):
#           for i in range(0, len(list_a), chunk_size):
#             yield list_a[i:i + chunk_size]
#     groups = list(split(bandit, group_size))
#     
#     values = []
#     #values = [round(x,2) for x in np.linspace(0,.5,len(groups))]
#     values = [round(.1*(val+1),1) for val in range(len(groups))]
#     opt = np.mean(sorted(values,reverse=True)[:K])
# =============================================================================
    opt = .25
    
    for _ in range(n_runs):
        rewards = etcg(bandit, K, T)[1]
        regret.append([opt-x for x in rewards])
        
    a = np.array(regret)
    return a   