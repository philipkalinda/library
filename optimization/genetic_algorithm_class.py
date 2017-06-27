# Method for genetic algorithm

# This genetic algorithm is used to select the best features within a standard regression model with the optimzation function focusing on maximizing the r2 value and minimizing the mean p-values

#Required Imports
import numpy as np
import time
from sklearn.model_selection import cross_val_score, KFold


class FeatureSelectionGeneticAlgorithm():
    """Built to be compatible with sci-kit learn library for both regression and classification models
    This is designed to help with feature selection in highly dimensional datasets
    """
    def __init__(self, mutation_rate = 0.001, iterations = 100, pool_size = 50):
        # imports
        # import numpy as np
        # from sklearn.model_selection import cross_val_score, KFold
        # self initiations
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.pool_size = pool_size
        self.pool = np.array([])
        self.iterations_results = {}
        self.kf = KFold(n_splits=5)

    def results(self):
        """Print best results from the fit
        """
        # return iterations_results[str(self.iterations+1)]['pool'][0]
        return self.pool[0]
        pass

    def fit(self, model, _type, X, y):
        """model = sci-kit learn regression/classification model
        X = X input data
        y = Y output data
        """
        # reset data in case run before (iterations_results hoted as a class variable and must be reset when refitting the Algorithm)

        # __init__(self.mutation_rate, self.iterations, self.pool_size)
        
        X = np.array(X)
        self.pool = np.random.randint(0,2,(self.pool_size, X.shape[1]))

        for iteration in range(1,self.iterations+1):
            s_t = time.time()
            scores = list(); fitness = list(); 
            for dna in self.pool:
                chosen_idx = [idx for bit, idx in zip(dna,range(X.shape[1])) if bit==1]
                adj_X = X[:,chosen_idx]

                if _type == 'regression':
                    score = np.mean(cross_val_score(model, adj_X, y, scoring='r2', cv=self.kf))

                elif _type == 'classification':
                    score = np.mean(cross_val_score(model, adj_X, y, scoring='accuracy', cv=self.kf))

                scores.append(score)
            fitness = [x/sum(scores) for x in scores]

            fitness, self.pool, scores = (list(t) for t in zip(*sorted(zip(fitness, [list(l) for l in list(self.pool)], scores),reverse=True)))

            # storage of iteration results
            self.iterations_results['{}'.format(iteration)] = dict()
            self.iterations_results['{}'.format(iteration)]['fitness'] = fitness
            self.iterations_results['{}'.format(iteration)]['pool'] = self.pool
            self.iterations_results['{}'.format(iteration)]['scores'] = scores

            self.pool = np.array(self.pool)

            if iteration != self.iterations+1:
                new_pool = []
                for dna in self.pool[1:int((len(self.pool)/2)+1)]:
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
            e_t = time.time()
            print('Iteration {} Complete [Time Taken: {} Seconds]'.format(iteration,round(e_t-s_t,2) ))
        
    