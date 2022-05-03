# -*- coding: utf-8 -*-

import numpy as np
import os as os; os.getcwd()
import random
#from random import seed
#import heapq

def dart(bandit, K, T):
    
    np.random.seed(int(random.uniform(0, 1000000)))
    
    "Defining DART class"
    class DART:
        def __init__(self, N, K, T, error_prob, env):
            self.N = N
            self.K = K
            self.T = T
            self.error_prob = error_prob
            
            self.environment = env

            self.precision = np.sqrt((N*np.log(N*T))/(T*K))
            
            print(self.N, self.K, self.T, self.precision)
        
        def __update_round_params__(self, r):
            delta = 1/np.power(2, r)
            nr = int(np.power(2, 2*r)*np.log(self.N*self.T))
            
            print(r, delta, nr)
            
            return delta, nr
        
        def __play_action__(self, action, t = 0):
            reward = self.environment.reward(action)
            if(t % (self.T/100) == 0):
                print(f"Finished {100*t / (self.T)}% of runs, current run {t}")
            return reward
            
        def find_best_K(self):
            t = 0
            A = []
            current_confused_arms = np.arange(self.N)
            R = []
            
            reward_list = []
            
            N = self.N
            K = self.K
            T = self.T

            #print(N, K)
            
            r = 1
            
            Delta, nr = self.__update_round_params__(r)
            
            num_plays = np.zeros(self.N)
            mu = np.zeros(self.N)
            
            num_arms = len(current_confused_arms)
            while(t < T):
        
                new_K = K-len(A)
                factor = (num_arms - new_K)/(num_arms-1)
                
                if( (num_arms + len(A) == 0) or
                    (T - t < np.ceil(num_arms/(K-len(A)))) ):
#                    print("breaking here after no arms left to explore")
#                    print(A)
#                    print(current_confused_arms)
#                    print(R)
#                    print(T, t, np.ceil(num_arms/(K-len(A))))
#                    print(r, Delta, nr)
                    break
                    
                np.random.shuffle(current_confused_arms)
    #             print(current_confused_arms)
    #             print(mu)
                
                for i in range(int(np.ceil(num_arms/(K-len(A))))):
                    action = []
                    action += A

                    if ((i+1)*(K-len(A)) <= num_arms):
                        action += current_confused_arms[i*(K-len(A)):(i+1)*(K-len(A))].tolist()
                    else:
                        action += current_confused_arms[num_arms - (K-len(A)):num_arms].tolist()

                    t += 1
                    reward_t = self.__play_action__(action, t)
                    reward_list.append(reward_t)
    #                 print(t, action, reward_t)

    #                 if ((i+1)*(K-len(A)) <= num_arms):
    #                     current_list = action
    #                 else:
    #                     current_list = action[i*(K-len(A)): num_arms]

                    num_plays[action] += 1
                    mu[action] = ((num_plays[action]-1)*mu[action]+(reward_t/factor))/num_plays[action]                

                sorted_index = np.argsort(-mu)
                last_K = mu[sorted_index[K-1]]
                top_K_1 = mu[sorted_index[K]]
                
                next_round_arms = np.ones(num_arms, dtype = bool)
                
                if(num_arms-new_K == 0):
                    print(num_arms, K, len(A))
                
                threshold = 2*Delta
    #             threshold = 2*((self.N - self.K)/(self.N-1))*Delta
    #             threshold = 2*Delta
                
                give_space = False
                for i in range(num_arms):
                    if(mu[current_confused_arms[i]] - top_K_1 > threshold):
                        A.append(current_confused_arms[i])
                        next_round_arms[i] = False

                        print(sorted_index, sorted_index[K])
                        print(top_K_1, threshold)
                        print(mu[current_confused_arms[i]], num_plays[current_confused_arms[i]])

                        print("accepted arms", threshold, A)
                        give_space = True
                    elif (last_K - mu[current_confused_arms[i]] > threshold):
                        R.append(current_confused_arms[i])
                        next_round_arms[i] = False
                        
                        print(sorted_index, sorted_index[K-1])
                        print(last_K, threshold)
                        print(mu[current_confused_arms[i]], num_plays[current_confused_arms[i]])
                        print("rejected arms", threshold, R)
                        give_space = True

                current_confused_arms = current_confused_arms[next_round_arms] 
                num_arms = len(current_confused_arms)
                        
                if(give_space):
                    print(mu)
                    print(num_plays, np.sum(num_plays), t)
                    print("current confused arms: ", current_confused_arms)
                    print("\n")
                                    
                assert(len(A) <= K)
                if( (len(A) + num_arms == K) or
                    (len(A) == K) ):
                    #print(mu)
                    #print(r, Delta, nr)
                    #print(A)
                    #print("breaking here after fill top K positions at time: ", t)
                    break
                            
                if( t > nr*np.ceil(num_arms/(K-len(A))) ):
#                     print(A)
#                     print(mu)

                    r += 1
                    Delta, nr = self.__update_round_params__(r)
        
                    if(Delta < self.precision):
                        #print("Delta got too low, breaking")
                        #print(Delta, self.precision, t, self.precision)
                        break
                    
            #select top K arms after exiting
            action = A + current_confused_arms[np.argsort(-mu[current_confused_arms])[0:K-len(A)]].tolist()
            
            while (t < T):
                t+=1
                reward_t = self.__play_action__(action, t)
                reward_list.append(reward_t)
                
                
            #print(mu)
            #print(num_plays)
            return reward_list, A, current_confused_arms

    "Defining MAB environment"
    class MAB_environment:
        def __init__(self, N, K):
            self.N = N
            self.K = K
            
        def reward(self, arms_list):
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

    "Performing DART"
    N = len(bandit)
    env = MAB_environment(N, K)
    error_prob = 1/(N*K*T)
    agent = DART(N, K, T, error_prob, env)  
    best_seed_sets_dart = []
    obs_influences_dart, _, N_set = agent.find_best_K()
    
    return best_seed_sets_dart, list(np.array(obs_influences_dart))

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
        rewards = dart(bandit, K, T)[1]
        regret.append([opt-x for x in rewards])
        
    a = np.array(regret)
    return a