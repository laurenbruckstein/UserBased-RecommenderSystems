import pandas as pd
import csv
from requests import get
import json
from datetime import datetime, timedelta, date
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr

import csv
import re
import pandas as pd
import argparse
import collections
import json
import glob
import math
import os
import requests
import string
import sys
import time
import xml
import random

class Recommender(object):
    def __init__(self, training_set, test_set):
        if isinstance(training_set, str):
            # the training set is a file name
            self.training_set = pd.read_csv(training_set)
        else:
            # the training set is a DataFrame
            self.training_set = training_set.copy()

        if isinstance(test_set, str):
            # the test set is a file name
            self.test_set = pd.read_csv(test_set)
        else:
            # the test set is a DataFrame
            self.test_set = test_set.copy()
    
    def train_user_euclidean(self, data_set, userId):

        # move userId col to very last col in data frame
        col = data_set.pop(userId)
        data_set.insert(len(data_set.columns), userId, col)
        sim_weights = {}
        for user in data_set.columns[1:-1]:
            data_set_subset = data_set[[userId,user]][data_set[userId].notnull() & data_set[user].notnull()] 
            if (data_set_subset.empty):
                sim_weights[user] = 0
            else:
                dist = euclidean(data_set_subset[userId], data_set_subset[user])
                sim_weights[user] = 1.0 / (1.0 + dist) 
        return sim_weights # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}
    
    def train_user_manhattan(self, data_set, userId):

        # move userId col to very last col in data frame
        col = data_set.pop(userId)
        data_set.insert(len(data_set.columns), userId, col)
        sim_weights = {}
        for user in data_set.columns[1:-1]:
            data_set_subset = data_set[[userId,user]][data_set[userId].notnull() & data_set[user].notnull()]
            if (data_set_subset.empty):
                sim_weights[user] = 0
            else:
                dist = cityblock(data_set_subset[userId], data_set_subset[user])
                sim_weights[user] = 1.0 / (1.0 + dist)
        return sim_weights # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}

    def train_user_cosine(self, data_set, userId): 
        sim_weights = {}
        for user in data_set.columns[1:]:
            if (user != userId):
                df_subset = data_set[[userId,user]][data_set[userId].notnull() & data_set[user].notnull()]
                if (df_subset.empty):
                    sim_weights[user] = 0
                else:
                    sim_weights[user] = cosine(df_subset[userId], df_subset[user])
        return sim_weights # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}
   
    def train_user_pearson(self, data_set, userId):

        # move userId col to very last col in data frame
        col = data_set.pop(userId)
        data_set.insert(len(data_set.columns), userId, col)
        sim_weights = {}
        for user in data_set.columns[1:-1]:
            df_subset = data_set[[userId,user]][data_set[userId].notnull() & data_set[user].notnull()]
            if(df_subset.empty):
                sim_weights[user] = 0
            else:
                p = pearsonr(df_subset[userId], df_subset[user])[0]
                if(p != np.nan):
                    sim_weights[user] = p
        return sim_weights  # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}

    def train_user(self, data_set, distance_function, userId):
        if distance_function == 'euclidean':
            return self.train_user_euclidean(data_set, userId)
        elif distance_function == 'manhattan':
            return self.train_user_manhattan(data_set, userId)
        elif distance_function == 'cosine':
            return self.train_user_cosine(data_set, userId)
        elif distance_function == 'pearson':
            return self.train_user_pearson(data_set, userId)
        else:
            return None

    def get_user_existing_ratings(self, data_set, userId):

        subset = data_set[['movieId', userId]][data_set['movieId'].notnull() & data_set[userId].notnull()]
        subset.values.tolist()
        tuples = [tuple(x) for x in subset.values]
        return tuples # list of tuples with movieId and rating. e.g. [(32, 4.0), (50, 4.0)]

    def predict_user_existing_ratings_top_k(self, data_set, sim_weights, userId, k):

        #sort sim_weights and take the top k
        sim_top = sorted(sim_weights.items(), key=lambda x:x[1], reverse=True)[:k]
        sim_top = dict(sim_top)

        #building predictions using sim_top
        predictions = []
        for index, row in data_set.iterrows():
            if (not np.isnan(row[userId])):
                predicted_rating = 0.0
                weights_sum = 0.0
                ratings = data_set.iloc[index][1:]
                for user in data_set.columns[1:]:
                    if (user != userId):
                        if (not np.isnan(ratings[user]) and user in sim_top.keys() and not np.isnan(sim_top[user])):
                            predicted_rating += ratings[user] * sim_top[user]
                            weights_sum += sim_top[user] 
                #if predicted_rating == 0 and weights_sum == 0:
                    #continue
                #else:
                    #predicted_rating /= weights_sum 
                    #predictions.append((row['movieId'], predicted_rating))
                if weights_sum != 0:
                    predicted_rating /= weights_sum 
                    predictions.append((row['movieId'], predicted_rating)) 


        return predictions # list of tuples with movieId and rating. e.g. [(32, 4.0), (50, 4.0)]
    
    def evaluate(self, existing_ratings, predicted_ratings):

        # Add all not nan and none values to new lists
        existing = []
        for ind, tupE in enumerate(existing_ratings):
            curRating = tupE[1]
            if curRating is not None and not math.isnan(curRating):
                existing.append(tupE)

        predicted = []
        for ind, tupP in enumerate(predicted_ratings):
            curRating = tupP[1]
            if curRating is not None and not math.isnan(curRating):
                predicted.append(tupP)

        # Calculate rmse
        mseSum = 0
        count = 0
        for indE, tupE in enumerate(existing):
            movieId = tupE[0]
            for indP, tupP in enumerate(predicted):
                if movieId == predicted[indP][0]:
                    mse = np.subtract(predicted[indP][1], existing[indE][1]) 
                    sq = np.square(mse)
                    mseSum = mseSum + sq
                    count = count + 1
                    break

        sqE = mseSum / count
        m = np.mean(sqE)
        rmse = math.sqrt(m)

        # Calculate ratio
        ratio = count / len(existing)
        return {'rmse':rmse, 'ratio':ratio} # dictionary with an rmse value and a ratio. e.g. {'rmse':1.2, 'ratio':0.5}
    
    def single_calculation(self, distance_function, userId, k_values):
        user_existing_ratings = self.get_user_existing_ratings(self.test_set, userId)
        print("User has {} existing and {} missing movie ratings".format(len(user_existing_ratings), len(self.test_set) - len(user_existing_ratings)), file=sys.stderr)

        print('Building weights')
        sim_weights = self.train_user(self.training_set[self.test_set.columns.values.tolist()], distance_function, userId)

        result = []
        for k in k_values:
            print('Calculating top-k user prediction with k={}'.format(k))
            top_k_existing_ratings_prediction = self.predict_user_existing_ratings_top_k(self.test_set, sim_weights, userId, k)
            result.append((k, self.evaluate(user_existing_ratings, top_k_existing_ratings_prediction)))
        return result # list of tuples, each of which has the k value and the result of the evaluation. e.g. [(1, {'rmse':1.2, 'ratio':0.5}), (2, {'rmse':1.0, 'ratio':0.9})]

    def aggregate_calculation(self, distance_functions, userId, k_values):
        print()
        result_per_k = {}
        for func in distance_functions:
            print("Calculating for {} distance metric".format(func))
            for calc in self.single_calculation(func, userId, k_values):
                if calc[0] not in result_per_k:
                    result_per_k[calc[0]] = {}
                result_per_k[calc[0]]['{}_rmse'.format(func)] = calc[1]['rmse']
                result_per_k[calc[0]]['{}_ratio'.format(func)] = calc[1]['ratio']
            print()
        result = []
        for k in k_values:
            row = {'k':k}
            row.update(result_per_k[k])
            result.append(row)
        columns = ['k']
        for func in distance_functions:
            columns.append('{}_rmse'.format(func))
            columns.append('{}_ratio'.format(func))
        result = pd.DataFrame(result, columns=columns)
        return result
        
if __name__ == "__main__":
    recommender = Recommender("data/train.csv", "data/small_test.csv")

    print("Training set has {} users and {} movies".format(len(recommender.training_set.columns[1:]), len(recommender.training_set)))
    print("Testing set has {} users and {} movies".format(len(recommender.test_set.columns[1:]), len(recommender.test_set)))

    result = recommender.aggregate_calculation(['euclidean', 'cosine', 'pearson', 'manhattan'], "0331949b45", [1, 2, 3, 4])
    #result = recommender.aggregate_calculation(['euclidean', 'cosine', 'pearson', 'manhattan'], "02839371ac", [1, 2, 3, 4])
    #result = recommender.single_calculation('euclidean', "02839371ac", [1])
    print(result)
