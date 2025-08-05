from collections import defaultdict
from surprise import Dataset, Reader, accuracy, KNNWithMeans
from APCF_surprise import APCF
from surprise.model_selection import PredefinedKFold
from sklearn.metrics import ndcg_score
import pandas as pd
from numpy.random import seed
from numpy import array, mean
import os

class RecAlgo:

    def __init__(self, name):
        self.name = name    # instance variable unique to each instance

    def get_metrics(self, predictions, k, threshold):
        """Return precision, recall and ndcg at k metrics for the dataset"""
                
        n_rec_t = 0
        n_rel_t = 0
        n_pred_t = 0
        ndcg = []
        n_imp = 0

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, details in predictions:
            user_est_true[uid].append((est, true_r))
            if(details["was_impossible"]):
                n_imp += 1

        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r > threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            ind_rec_k = [(est > threshold) for (est, _) in user_ratings[:k]]
            n_rec_k = sum(ind_rec_k)

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(
                ((true_r > threshold) and (est > threshold))
                for (est, true_r) in user_ratings[:k]
            )
            
            n_rec_t += n_rec_k
            n_rel_t += n_rel
            n_pred_t += n_rel_and_rec_k
            
            if(n_rec_k > 1):
                top_k = array(user_ratings[:k])[ind_rec_k]
                y_score, y_true = [[i] for i in zip(*top_k)]
                ndcg.append(ndcg_score(y_true, y_score))
            
        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precision = n_pred_t / n_rec_t if n_rec_t != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        
        recall = n_pred_t / n_rel_t if n_rel_t != 0 else 0
        
        n_pred = len(predictions)
        coverage = (n_pred - n_imp)/n_pred
        
        return precision, recall, mean(ndcg), coverage
        
    def precision_recall_at_k(self, predictions, k, threshold):
        """Return precision and recall at k metrics for each user"""

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls
        
    def run_KFold(self, n_splits, files_dir, dataset, k=10, threshold=2.5, sd=1, save_pred=False, scale=(1,5)):
    
        # files_dir: path to dataset folder
        # dataset: dataset name
        # k: parameter for precision @ k
        # threshold: rating relevance limit
        # sd: seed for random number generator
        
        seed(sd)

        reader = Reader(line_format="user item rating", sep=";", skip_lines=1, rating_scale=scale)

        # folds_files is a list of tuples containing file paths:
        # [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
        train_file = files_dir + dataset + '_train%d.csv'
        test_file = files_dir + dataset + '_test%d.csv'
        folds_files = [(train_file % i, test_file % i) for i in range(n_splits)]

        data = Dataset.load_from_folds(folds_files, reader=reader)
        pkf = PredefinedKFold()

        algo = self.algo

        acc = []
        prec = []
        rec = []
        ndcg = []
        coverage = []
        
        i = 0

        for trainset, testset in pkf.split(data):
        
            print(f'Running {self.name} on {dataset} - fold{i+1}')
        
            algo.fit(trainset)
            predictions = algo.test(testset)
            
            accur = accuracy.mae(predictions, verbose=True)
            acc.append(accur)
            #precisions, recalls = precision_recall_at_k(predictions, k, threshold)
            precision, recall, dcg, cvg = self.get_metrics(predictions, k, threshold)
            
            if(save_pred):
                df_pred = pd.DataFrame(predictions, columns=['uid','iid','r_ui','est','details'])
                df_pred.to_csv(files_dir + dataset + '_' + self.name + f'_pred{i}.csv', index=False)
            
            prec.append(precision)
            rec.append(recall)
            ndcg.append(dcg)
            coverage.append(cvg)

            print(f'Precision: {precision}, Recall: {recall}, NDCG: {dcg}, Coverage: {cvg}')
            
            i += 1
            
        df = pd.DataFrame(data={'Accuracy' : acc, 'Precision' : prec, 'Recall' : rec, 'NDCG' : ndcg, 'Coverage': coverage})
        df.to_csv(files_dir + dataset + '_' + self.name + '_results.csv', index=False)
              
class RecAPCF(RecAlgo):

    def __init__(self, K, n, sim_name, optimize_sim, n_ass):
        self.name = 'APCF'
        sim_options = {
            "name": sim_name,
            "user_based": False,  # compute  similarities between items
        }
        algo = APCF(k=K, n_pat = n, sim_options=sim_options, optimize_sim=optimize_sim, n_ass=n_ass)
        self.algo = algo
        
    def run_KFold(self, n_splits, files_dir, dataset, k=25, t_rat=2.5, t_bin = 0, sd=1, save_pred=False, scale=(1,5), save_metrics=True):
    
        # files_dir: path to dataset folder
        # dataset: dataset name
        # k: parameter for precision @ k
        # t_rat: rating relevance limit
        # t_bin: binarization threshold
        # sd: seed for random number generator
        # save_pred: save predictions
        
        seed(sd)

        reader = Reader(line_format="user item rating", sep=";", skip_lines=1, rating_scale=scale)

        # folds_files is a list of tuples containing file paths:
        # [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
        train_file = files_dir + dataset + '_train%d.csv'
        test_file = files_dir + dataset + '_test%d.csv'
        pat_file = files_dir + dataset + '_fold%d_t'+ f'{int(t_bin*10)}.binaps.patterns'
        
        folds_files = []
        pat_files = []
        
        for i in range(n_splits):
        
            folds_files += [(train_file % i, test_file % i)]
            pat_files += [(pat_file % i)]
        
        data = Dataset.load_from_folds(folds_files, reader=reader)
        pkf = PredefinedKFold()

        algo = self.algo

        acc = []
        prec = []
        rec = []
        ndcg = []
        coverage = []

        for i, (trainset, testset) in enumerate(pkf.split(data)):
            
            print(f'Running {self.name} on {dataset} - fold{i+1}')
            
            algo.fit(trainset, pat_files[i])
            predictions = algo.test(testset)
            
            print("Max number of similar items: ", algo.max_items)
            print("Time for user similarities: ", algo.time_user)
            print("Time for neighborhood: ", algo.time_neigh)

            accur = accuracy.mae(predictions, verbose=True)
            acc.append(accur)
            #precisions, recalls = precision_recall_at_k(predictions, k, t_rat)
            precision, recall, dcg, cvg = self.get_metrics(predictions, k, t_rat)
            
            if(save_pred):
                df_pred = pd.DataFrame(predictions, columns=['uid','iid','r_ui','est','details'])
                df_pred.to_csv(files_dir + dataset + '_' + self.name + f'_pred{i}.csv', index=False)
            
            prec.append(precision)
            rec.append(recall)
            ndcg.append(dcg)
            coverage.append(cvg)

            print(f'Precision: {precision}, Recall: {recall}, NDCG: {dcg}, Coverage: {cvg}')
        
        if(save_metrics):    
            df = pd.DataFrame(data={'Accuracy' : acc, 'Precision' : prec, 'Recall' : rec, 'NDCG' : ndcg, 'Coverage': coverage})
            df.to_csv(files_dir + dataset + '_' + self.name + '_results.csv', index=False)
        
        return acc, prec, rec, ndcg, coverage

