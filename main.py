# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
from tqdm import tqdm

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

movies_with_ratings = movies.merge(ratings, on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)

num_users = movies_with_ratings.userId.unique().shape[0]

movie_vector = {}

for movie, group in tqdm(movies_with_ratings.groupby('title')):
    movie_vector[movie] = np.zeros(num_users)
    
    for i in range(len(group.userId.values)):
        u = group.userId.values[i]
        r = group.rating.values[i]
        movie_vector[movie][int(u - 1)] = r
        
        
        
        #  Порекомендуйте похожие фильмы по мотивам ранее обученной модели. Шаги: 3
def showSimilarMovies(algo, rid_to_name, name_to_rid):
    #  Получить raw_id фильма История игрушек (1995)
    toy_story_raw_id = name_to_rid['Toy Story (1995)']
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
    #Получить рекомендуемые фильмы через модель здесь 10
    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, 10)
    #Преобразовать внутренний идентификатор модели в фактический идентификатор фильма
    neighbors_raw_ids = [algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors]
    #По списку идентификаторов фильмов или списку рекомендаций по фильмам
    neighbors_movies = [rid_to_name[raw_id] for raw_id in neighbors_raw_ids]
    print('The 10 nearest neighbors of Toy Story are:')
    for movie in neighbors_movies:
        print(movie)