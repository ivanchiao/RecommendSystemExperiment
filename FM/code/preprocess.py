# -*- coding: utf-8 -*-
"""
@author

██╗     ███╗   ███╗ ██████╗      ███████╗ ██████╗
██║     ████╗ ████║██╔════╝      ╚══███╔╝██╔════╝
██║     ██╔████╔██║██║     █████╗  ███╔╝ ██║
██║     ██║╚██╔╝██║██║     ╚════╝ ███╔╝  ██║
███████╗██║ ╚═╝ ██║╚██████╗      ███████╗╚██████╗
╚══════╝╚═╝     ╚═╝ ╚═════╝      ╚══════╝ ╚═════╝
"""

import numpy as np
import pandas as pd


class Preprocess(object):

    def __init__(self, root_path, ratio=0.8):

        self.root_path = root_path
        self.ratio = ratio

        self.users_name = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
        self.movies_name = ["MovieID", "Title", "Genres"]
        self.ratings_name = ["UserID", "MovieID", "Rating", "Timestamp"]
        self.usecol = ["UserID", "MovieID", "Gender", "Age", "Occupation", 'Genres',
                       'Rating']  # eliminate zip-code, timestamp, title

        self.sparse_n_features = 0  # 稀疏向量(user_id, movie_id)的维度
        self.dense_n_features = 0  # 密集向量的维度

        self.data = self.load()
        self.split_data()

    def __len__(self):
        return len(self.data)

    def load(self):
        """
        以 {UserID, MovieID}为主键连接数据集

        """

        users = pd.read_csv(self.root_path + "movielens-1m/raw/users.dat", sep="::",
                            engine='python', names=self.users_name)
        movies = pd.read_csv(self.root_path + "movielens-1m/raw/movies.dat", sep="::",
                             engine='python', names=self.movies_name)
        ratings = pd.read_csv(self.root_path + "movielens-1m/raw/ratings.dat", sep="::",
                              engine='python', names=self.ratings_name)

        data = pd.merge(users, pd.merge(movies, ratings, on="MovieID"),
                        on="UserID")
        data = data[self.usecol]

        return data

    def feature_engineering(self):

        user_id = self.user_id_pro()
        item_id = self.item_id_pro()
        gender = self.gender_pro()
        age = self.age_pro()
        occupation = self.occupation_pro()
        genres = self.genres_pro()
        ratings = self.rating_pro()

        user_id = np.expand_dims(user_id, axis=1)
        item_id = np.expand_dims(item_id, axis=1)
        ratings = np.expand_dims(ratings, axis=1)
        dataset = np.concatenate((user_id, item_id, gender, age, occupation, genres, ratings), axis=1)

        return dataset

    def user_id_pro(self):

        self.sparse_n_features += self.data['UserID'].max()

        user_id = self.data['UserID'].to_numpy()
        user_id = user_id - 1
        return np.float32(user_id)

    def item_id_pro(self):
        self.sparse_n_features += self.data['MovieID'].max()

        item_id = self.data['MovieID'].to_numpy()
        item_id = item_id - 1

        return np.float32(item_id)

    def gender_pro(self):
        self.dense_n_features += 2

        gender = self.data['Gender'].to_numpy()
        gender = np.int32(gender == 'M')
        one_hot = np.eye(2, dtype=np.float32)[gender.tolist()]
        return one_hot

    def age_pro(self):
        MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
        self.dense_n_features += len(MAP)
        age = list(map(lambda x: MAP[x], self.data['Age'].tolist()))
        one_hot = np.eye(len(MAP), dtype=np.float32)[age]

        return one_hot

    def occupation_pro(self):

        occupation_max = self.data['Occupation'].max() + 1
        self.dense_n_features += occupation_max
        one_hot = np.eye(occupation_max, dtype=np.float32)[self.data['Occupation'].tolist()]

        return one_hot

    def genres_pro(self):
        MAP = {'Action': 0,
               'Adventure': 1,
               'Animation': 2,
               'Children\'s' : 3,
               'Comedy' : 4,
               'Crime' : 5,
               'Documentary': 6,
               'Drama': 7,
               'Fantasy': 8,
               'Film-Noir': 9,
               'Horror': 10,
               'Musical': 11,
               'Mystery': 12,
               'Romance': 13,
               'Sci-Fi': 14,
               'Thriller': 15,
               'War': 16,
               'Western': 17}

        self.dense_n_features += len(MAP)
        one_hot = np.zeros((self.__len__(), len(MAP)), dtype=np.float32)
        for i, vals in enumerate(self.data['Genres']):

            for val in vals.split('|'):
                j = MAP[val]
                one_hot[i, j] = 1
        return one_hot

    def rating_pro(self):

        rank = [0, 0, 0, 0 ,0]

        for i in self.data['Rating'].to_numpy():
            rank[i-1] += 1

        for i in rank:
            print(i)

        ratings = self.data['Rating'].to_numpy()

        return np.float32(ratings)

    def split_data(self):
        n_train = int(self.__len__() * self.ratio)

        dataset_process = self.feature_engineering()
        np.random.shuffle(dataset_process)

        train_set = dataset_process[:n_train, :]
        test_set = dataset_process[n_train:, :]

        np.savez(self.root_path + 'movielens-1m/processed/movielens-1m', train_set=train_set, test_set=test_set,
                 sparse_n_features=self.sparse_n_features, dense_n_features=self.dense_n_features)


if __name__ == '__main__':
    d = Preprocess('../data/')