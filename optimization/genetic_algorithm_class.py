# Method for genetic algorithm

# This genetic algorithm is used to select the best features within a standard regression model with the optimzation function focusing on maximizing the r2 value and minimizing the mean p-values

#Required Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold


class FeatureSelectionGeneticAlgorithm:
    """Built to be compatible with sci-kit learn library for both regression and classification models
    This is designed to help with feature selection in highly dimensional datasets
    """
    def __init__(self,mutation_rate = 0.001, iterations = 250, pool_size = 50):
        # imports
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import cross_val_score, KFold
        # self initiations
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.pool_size = pool_size
        self.pool = np.array([])
        self.iterations_results = {}
        self.kf = KFold(n_splits=5)

    def results(self):
        return self.pool[0]
        pass

    def fit(self, model, _type, X, y):
        """model = sci-kit learn regression/classification model
        X = X input data
        y = Y output data
        """
        # reset data in case run before (iterations_results hoted as a class variable and must be reset when refitting the Algorithm)
        __init__(self.mutation_rate, self.iterations, self.pool_size)
        X = np.array(X)
        # - legacy - # self.pool = [[np.random.randint(0,2) for genes in range(X.shape[1])] for dna in range(self.pool_size)]
        self.pool = np.random.randint(0,2,(self.pool_size, X.shape[1]))

        for iteration in range(1,self.iterations+1):
            scores = list(); fitness = list(); 
            for dna in self.pool:
                chosen_idx = [idx for bit, idx in zip(dna,range(X.shape[1])) if bit==1]
                adj_X = X[:,chosen_idx]

                if _type = 'regression':
                    score = np.mean(cross_val_score(model, X, y, scoring='r2', cv=self.kf))

                elif _type = 'classification':
                    score = np.mean(cross_val_score(model, X, y, scoring='accuracy', cv=self.kf))

                scores.append(score)
            fitness = [x/sum(scores) for x in scores]

            fitness, self.pool, scores = (list(t) for t in zip(*sorted(zip(fitness, self.pool, scores),reverse=True)))

            # storage of iteration results
            self.iterations_results['{}'.format(iteration)] = dict()
            self.iterations_results['{}'.format(iteration)]['fitness'] = fitness
            self.iterations_results['{}'.format(iteration)]['pool'] = self.pool
            self.iterations_results['{}'.format(iteration)]['scores'] = scores


            if iteration != self.iterations+1:
                new_pool = []
                for dna in self.pool[1:,int((len(self.pool)/2)+1)]:
                    random_split_point = np.random.randint(1,len(dna))
                    next_gen1 = np.concatenate((self.pool[0][:random_split_point], dna[random_split_point:]), axis = 0)
                    next_gen2 = np.concatenate((dna[:random_split_point], self.pool[0][random_split_point:]), axis = 0)
                    for idx, chromosome in enumerate(next_gen1):
                        if np.random.random() < self.mutation_rate:
                            next_gen1[idx] = 1 if chromosome==0 else 0
                    for idx, chromosome in enumerate(next_gen2):
                        if np.random.random() < self.mutation_rate:
                            next_gen2[idx] = 1 if chromosome==0 else 0
                    new_pool.append(next_gen1)
                    new_pool.append(next_gen2)
                self.pool = new_pool
            else:
                continue
        
    







# def genetic_algorithm(xdata, ydata, mutation_rate=0.001, iterations=10):    
#     pool = [[np.random.randint(0,2) for genes in range(2000)] for dna in range(50)]
#     iterations_results = {}
    
#     for iteration in range(1,iterations+1):
#         print('Epoch {}:'.format(iteration))
#         st  = time.time()
        
#         #################################################################################
#         scores = list(); r_squared_adj = list(); mean_pvalues = list(); fitness = list()
        
#         for dna in pool:
#             chosen_cols = [col for identity,col in zip(dna,xdata.iloc[:,20:].columns.tolist()) if identity ==1]
#             X = pd.concat([xdata.iloc[:,:20],xdata.loc[:,chosen_cols]],axis=1)
            
#             sm_X_data = sm.add_constant(X)
#             results = sm.OLS(ydata,sm_X_data).fit()
            
#             score = results.rsquared_adj / np.mean(results.pvalues)
            
#             r_squared_adj.append(results.rsquared_adj)
#             mean_pvalues.append(np.mean(results.pvalues))
#             scores.append(score)
#         fitness = [x/sum(scores) for x in scores]
#         end = time.time() #Time Edit Note - Change the time location
        
#         #################################################################################
#         print('\tTime Taken: {}'.format(end-st))
#         print('\tBest Score: {}'.format(max(scores)))
#         print('\tBest r2: {}'.format(r_squared_adj[scores.index(max(scores))]))
#         print('\tBest mean pvalues: {}'.format(mean_pvalues[scores.index(max(scores))]))
        
#         #################################################################################
#         fitness, pool, scores, r_squared_adj, mean_pvalues = (list(t) for t in zip(*sorted(zip(fitness, pool, scores, r_squared_adj, mean_pvalues),reverse=True)))
        
#         #################################################################################
#         iterations_results['{}'.format(iteration)] = dict()
#         iterations_results['{}'.format(iteration)]['fitness'] = fitness
#         iterations_results['{}'.format(iteration)]['pool'] = pool
#         iterations_results['{}'.format(iteration)]['scores'] = scores
#         iterations_results['{}'.format(iteration)]['r_squared_adj'] = r_squared_adj
#         iterations_results['{}'.format(iteration)]['mean_pvalues'] = mean_pvalues
        
#         #################################################################################
#         if iteration != iterations+1:
#             new_pool = []
#             for dna in pool[1:int((len(pool)/2)+1)]:
#                 random_split_point = np.random.randint(1,len(dna))
#                 new_dna1 = pool[0][:random_split_point]+dna[random_split_point:]
#                 new_dna2 = dna[:random_split_point]+pool[0][random_split_point:]
#                 for idx,chromosome in enumerate(new_dna1):
#                     if np.random.random() <mutation_rate:
#                         new_dna1[idx] = 1 if chromosome==0 else 0
#                 for idx,chromosome in enumerate(new_dna2):
#                     if np.random.random() <mutation_rate:
#                         new_dna2[idx] = 1 if chromosome==0 else 0
#                 new_pool.append(new_dna1)
#                 new_pool.append(new_dna2)
#             pool = new_pool.copy()
#         else:
#             print('Genetic Algorithm Complete')
        
#         #################################################################################
        
#     best_cols = [col for identity,col in zip(pool[0],xdata.iloc[:,20:].columns.tolist()) if identity ==1]
#     best_X = pd.concat([xdata.iloc[:,:20],xdata.loc[:,best_cols]],axis=1)
#     sm_X_data = sm.add_constant(best_X)
#     best_results = sm.OLS(Y_c_data,sm_X_data).fit()
        
#     return iterations_results, pool[0], best_cols, best_X, best_results
