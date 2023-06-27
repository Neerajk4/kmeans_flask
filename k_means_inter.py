
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from random import uniform
from sklearn.datasets import load_iris


def initialize_centroids(X: np.ndarray, k: int) -> dict:
    #initialies random centroids
    centroid_dict = {}
    min_x, max_x = np.min(X, axis=0), np.max(X, axis=0)
    for i in range(k):
        centroid_dict[i] = uniform(min_x, max_x)
    #print('initialize_centroids')
    return centroid_dict

def assign_clusters(X: np.ndarray, df:pd.core.frame.DataFrame, centroids: dict):
    #appends centroid of cluster closest to feature set.
    cluster_list = []
    for featureset in X:
        distances = [np.linalg.norm(featureset-centroids[centroid]) for centroid in centroids]
        classification = distances.index(min(distances))
        cluster_list.append(classification)
    df.loc[:, 'cluster'] = cluster_list
    #print('assign_clusters')
    return df

def reclassify_centroids(df:pd.core.frame.DataFrame, centroids: dict, column_names: list):
    #reclassifying centroids
    for c in centroids:
        x1 = np.average(df[df['cluster'] == c][column_names[0]])
        y1 = np.average(df[df['cluster'] == c][column_names[1]])
        centroids[c] = np.array([x1, y1])
    return centroids

def optimize_check(centroids:dict, prev_centroids: dict, tol: float):
    optimized = True
    # Checks if centroids are optimized.  If so breaks the loop
    for c in centroids:
        original_centroid = prev_centroids[c]
        current_centroid = centroids[c]
        if np.absolute(np.sum((current_centroid-original_centroid)/original_centroid*100.0)) > tol:
            optimized = False
    return optimized


def kmeans_iter(k: int, tol: float, max_iter: int, column_names: list):
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = pd.Series(data.target)
    df = df[column_names]
    X = np.array(df)
    centroids = initialize_centroids(X, k)

    fig, ax = plt.subplots(3, 2, figsize=(50, 40))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'gainsboro', 'mistyrose', 'yellow', 'aqua', 'blue']

    # iterates through iterations
    for j in range(max_iter):
        df = assign_clusters(X, df, centroids)
        prev_centroids = dict(centroids)
        centroids = reclassify_centroids(df, centroids, column_names)
        optimized = optimize_check(centroids, prev_centroids, tol)

        if optimized:
            break

        row_num = j // 2
        col_num = j % 2

        for value in centroids:
            df2 = df[df['cluster'] == value]
            ax[row_num, col_num].scatter(df2[column_names[0]], df2[column_names[1]], c=colors[value],
                                         label='cluster ' + str(value), edgecolors='none')
            ax[row_num, col_num].scatter(centroids[value][0], centroids[value][1], c=colors[value], s=400, marker='+')
            ax[row_num, col_num].set_xlabel(column_names[0])
            ax[row_num, col_num].set_ylabel(column_names[1])
            ax[row_num, col_num].set_title('Iteration number ' + str(j), fontsize = 20)
            ax[row_num, col_num].legend()
            ax[row_num, col_num].grid(True)

    #plt.show()
    fig.savefig('static/uploads/kmeans_iteration.png')






