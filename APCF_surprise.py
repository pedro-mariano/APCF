#import pyximport
#pyximport.install(setup_args={"script_args" : ["--verbose"]})

import heapq
import numpy as np
from surprise.prediction_algorithms.knns import SymmetricAlgo
from surprise import PredictionImpossible
from surprise import similarities as sims
from APCF_sims import adj_cosine, sim_user_pat
import timeit

class APCF(SymmetricAlgo):

    def __init__(self, k=40, min_k=1, n_pat = 10, sim_options={}, verbose=True, optimize_sim =False, n_ass = 1, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.n_pat = n_pat
        self.optimize_sim = optimize_sim # If true, compute similarity only for items in the pattern set
        self.n_ass = n_ass # Number of similar users for pattern selection
        
        self.time_user = 0
        self.time_neigh = 0
        self.max_items = 0

    def fit(self, trainset, pat_file):

        SymmetricAlgo.fit(self, trainset)
        self.read_patterns(pat_file)
        print('Number of patterns before selection: ', len(self.patterns))
        self.pattern_selection()
        print('Number of patterns after selection: ', len(self.patterns))
        
        self.u_means = np.zeros(self.n_y)
        self.i_means = np.zeros(self.n_x)
        
        for y, ratings in self.yr.items():
            self.u_means[y] = np.mean([r for (_, r) in ratings])
            
        for x, ratings in self.xr.items():
            self.i_means[x] = np.mean([r for (_, r) in ratings])
        
        self.sim = self.compute_similarities(use_patterns=self.optimize_sim)
        #self.sim = self.compute_similarities()
        self.item_set = {} # Dict containing the item-set for each user

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")

        x, y = self.switch(u, i)
        
        # Read or build the item set containing the items from the n_pat most similar patterns to target user
        if(u in self.item_set):
            item_set = self.item_set[u]
        else:
            
            start = timeit.default_timer()

            sim_pats = [(p, self.sim_user[i,u]) for (i, p) in enumerate(self.patterns)]
            n_sim_pats = heapq.nlargest(self.n_pat, sim_pats, key=lambda t: t[1])
            item_set = np.unique([i for sim in n_sim_pats for i in sim[0]])
            
            self.item_set[u] = item_set
           
            self.time_user += timeit.default_timer() - start
            
        start = timeit.default_timer()

        # Select the k nearest neighbors 
        #neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y] if x2 in item_set]

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        eval_items = [i[0] for i in neighbors]
        pool_items = set(eval_items).intersection(item_set)
        neighbors = [n for n in neighbors if n[0] in pool_items]
        
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: abs(t[1]))
        
        #k_neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y] if x2 in item_set and self.sim[x, x2] > 0]
        
        self.time_neigh += timeit.default_timer() - start

        mean_user = self.u_means[u]
        
        #sim_items = [n for n in neighbors if n[1] > 0]
        #self.max_items = max(self.max_items, len(sim_items))
        self.max_items = max(self.max_items, len(k_neighbors))

        '''
        if(len(sim_items) == 0):
            mean_user = self.u_means[u]
        else:
            mean_user = np.mean([n[2] for n in sim_items])
        '''
        est = mean_user

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            sum_sim += abs(sim)
            sum_ratings += sim * (r - mean_user)
            actual_k += 1

        details = {}

        if actual_k < self.min_k:
            sum_ratings = 0
            
        details["actual_k"] = actual_k

        if(sum_sim != 0):
            est += sum_ratings / sum_sim
            details["no_neighbors"] = False
            
        else:
            details["no_neighbors"] = True

        return est, details
        
    def compute_similarities(self, use_patterns=False):
        """Build the similarity matrix.

        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        Returns:
            The similarity matrix."""

        construction_func = {
            "cosine": sims.cosine,
            "msd": sims.msd,
            "pearson": sims.pearson,
            "pearson_baseline": sims.pearson_baseline,
            "adj_cosine": adj_cosine
        }

        if self.sim_options["user_based"]:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get("min_support", 1)
        
        if(use_patterns):
            item_set = np.unique([i for p in self.patterns for i in p])
            print('Item-set size: ', len(item_set))
            for u, ratings in yr.items():
                yr[u] = [i for i in ratings if i[0] in item_set] # Only ratings in the item-set

        args = [n_x, yr, min_support]

        name = self.sim_options.get("name", "msd").lower()
        if name == "pearson_baseline":
            shrinkage = self.sim_options.get("shrinkage", 100)
            bu, bi = self.compute_baselines()
            if self.sim_options["user_based"]:
                bx, by = bu, bi
            else:
                bx, by = bi, bu

            args += [self.trainset.global_mean, bx, by, shrinkage]
        elif name == "adj_cosine":
            args += [self.u_means]
                
        try:
            if getattr(self, "verbose", False):
                print(f"Computing the {name} similarity matrix...")
                
                
            start = timeit.default_timer()    
            
            sim = construction_func[name](*args)
            
            ts = timeit.default_timer() - start
            
            if getattr(self, "verbose", False):
                print("Done computing similarity matrix.")
                print("Time to compute similarity: ", ts)
            return sim
        except KeyError:
            raise NameError(
                "Wrong sim name "
                + name
                + ". Allowed values "
                + "are "
                + ", ".join(construction_func.keys())
                + "."
            )
            
    def read_patterns(self, filepath):
        # Returns list of patterns in filepath

        patterns = []
        with open(filepath, 'r') as f:
            array = ''
            for line in f:
                if(line[-2] != ']'):
                    array += line[1:-1]+' '
                    continue
                else:
                    array += line[1:-2]
                    cols = array.split()
                    pat = [self.trainset.to_inner_iid(i) for i in cols]
                    patterns.append(pat)
                    array = ''
                
        self.patterns = patterns
        
        # Calculate user-pattern similarity 
        self.sim_user = sim_user_pat(self.n_y, patterns, self.xr)
        
    def pattern_selection(self, t_spar=None):

        # Select patterns based on sparsity or number of associated users
        # t_spar: minimum sparsity
        
        n_users = self.trainset.n_users
        n_items = self.trainset.n_items
        ir = self.trainset.ir
        
        sel_patterns = []

        if(t_spar != None):
            # If sparsity threshold is provided, select patterns above threshold
            
            # Calculate the data sparsity for each pattern set
            
            ni_ratings = np.zeros(n_items, dtype=int)
        
            for i, ratings in ir.items():
                ni_ratings[i] = len(ratings)
            
            for p in self.patterns:
                np_ratings = np.sum(ni_ratings[p])
                sparsity = 100*np_ratings/(n_users * len(p))
                if(sparsity > t_spar):
                    sel_patterns.append(p)
        else:
            # Otherwise, select patterns associated with more than n_ass similar users
        
            sel_index = []
            for i, p in enumerate(self.patterns):
                
                na_users = np.count_nonzero(self.sim_user[i] > 0.5)
                if(na_users > self.n_ass):
                    sel_patterns.append(p)
                    sel_index.append(i)
        
        self.patterns = sel_patterns
        self.sim_user = self.sim_user[sel_index]

