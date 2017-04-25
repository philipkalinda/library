# Method for genetic algorithm

# This genetic algorithm is used to select the best features within a standard regression model with the optimzation function focusing on maximizing the r2 value and minimizing the mean p-values

import numpy as np
import statsmodels.api as sm


def genetic_algorithm(xdata, ydata, mutation_rate=0.001, iterations=10):    
    pool = [[np.random.randint(0,2) for genes in range(2000)] for dna in range(50)]
    iterations_results = {}
    
    for iteration in range(1,iterations+1):
        print('Epoch {}:'.format(iteration))
        st  = time.time()
        
        #################################################################################
        scores = list(); r_squared_adj = list(); mean_pvalues = list(); fitness = list()
        
        for dna in pool:
            chosen_cols = [col for identity,col in zip(dna,xdata.iloc[:,20:].columns.tolist()) if identity ==1]
            X = pd.concat([xdata.iloc[:,:20],xdata.loc[:,chosen_cols]],axis=1)
            
            sm_X_data = sm.add_constant(X)
            results = sm.OLS(ydata,sm_X_data).fit()
            
            score = results.rsquared_adj / np.mean(results.pvalues)
            
            r_squared_adj.append(results.rsquared_adj)
            mean_pvalues.append(np.mean(results.pvalues))
            scores.append(score)
        fitness = [x/sum(scores) for x in scores]
        end = time.time() #Time Edit Note - Change the time location
        
        #################################################################################
        print('\tTime Taken: {}'.format(end-st))
        print('\tBest Score: {}'.format(max(scores)))
        print('\tBest r2: {}'.format(r_squared_adj[scores.index(max(scores))]))
        print('\tBest mean pvalues: {}'.format(mean_pvalues[scores.index(max(scores))]))
        
        #################################################################################
        fitness, pool, scores, r_squared_adj, mean_pvalues = (list(t) for t in zip(*sorted(zip(fitness, pool, scores, r_squared_adj, mean_pvalues),reverse=True)))
        
        #################################################################################
        iterations_results['{}'.format(iteration)] = dict()
        iterations_results['{}'.format(iteration)]['fitness'] = fitness
        iterations_results['{}'.format(iteration)]['pool'] = pool
        iterations_results['{}'.format(iteration)]['scores'] = scores
        iterations_results['{}'.format(iteration)]['r_squared_adj'] = r_squared_adj
        iterations_results['{}'.format(iteration)]['mean_pvalues'] = mean_pvalues
        
        #################################################################################
        if iteration != iterations+1:
            new_pool = []
            for dna in pool[1:int((len(pool)/2)+1)]:
                random_split_point = np.random.randint(1,len(dna))
                new_dna1 = pool[0][:random_split_point]+dna[random_split_point:]
                new_dna2 = dna[:random_split_point]+pool[0][random_split_point:]
                for idx,chromosome in enumerate(new_dna1):
                    if np.random.random() <0.001:
                        new_dna1[idx] = 1 if chromosome==0 else 0
                for idx,chromosome in enumerate(new_dna2):
                    if np.random.random() <0.001:
                        new_dna2[idx] = 1 if chromosome==0 else 0
                new_pool.append(new_dna1)
                new_pool.append(new_dna2)
            pool = new_pool.copy()
        else:
            print('Genetic Algorithm Complete')
        
        #################################################################################
        
    best_cols = [col for identity,col in zip(pool[0],xdata.iloc[:,20:].columns.tolist()) if identity ==1]
    best_X = pd.concat([xdata.iloc[:,:20],xdata.loc[:,best_cols]],axis=1)
    sm_X_data = sm.add_constant(best_X)
    best_results = sm.OLS(Y_c_data,sm_X_data).fit()
        
    return iterations_results, pool[0], best_cols, best_X, best_results
