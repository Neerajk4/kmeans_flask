

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets


def kmeans_run(df, cluster_num):
    kmeans_cluster = KMeans(n_clusters=cluster_num)
    x = df.iloc[:, [0, 1, 2, 3]].values
    y_kmeans = kmeans_cluster.fit_predict(x)
    df['cluster_num'] = y_kmeans
    df['cluster_num'] = df['cluster_num'].astype(str)
    return df

def kmeans_plot(df, cluster_number, column1_name, column2_name):
    fig, ax = plt.subplots()
    flatui = sns.color_palette("pastel", cluster_number)
    b = sns.scatterplot(ax=ax, x=column1_name, y=column2_name, data=df,
    hue=df['cluster_num'], palette=flatui)
    ax.set_xlabel(column1_name)
    ax.set_ylabel(column2_name)
    fig.savefig('static/uploads/kmeans.png')

def cluster_main(column1_name, column2_name, cluster_number):
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    df = kmeans_run(df, cluster_number)
    kmeans_plot(df, cluster_number, column1_name, column2_name)