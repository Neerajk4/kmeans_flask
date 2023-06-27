# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 20:01:20 2022

@author: Neerajk4
"""

##%
import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
##from sklearn.externals import joblib
import re
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import altair as alt

#%%

os.chdir("Documents/projects/kmeans_flask")


#%%

cluster_number = 4


iris = datasets.load_iris()

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
x = df.iloc[:, [0, 1, 2, 3]].values

column1_name = 'sepal length (cm)'
column2_name = 'petal length (cm)'

kmeans_cluster = KMeans(n_clusters=cluster_number)
y_kmeans = kmeans_cluster.fit_predict(x)

df['cluster_num'] = y_kmeans
df['cluster_num'] = df['cluster_num'].astype(str)
column_list = df.columns
img = BytesIO()

#%%

plt.scatter(x[:, 0], x[:, 3], c=y_kmeans5, cmap='rainbow')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[3])
    ##plt.legend(loc='lower right')
    
#%%
fig, ax = plt.subplots(figsize=(30, 20))
plt.scatter(df[column1_name], df[column2_name], c=y_kmeans5, cmap='rainbow')
plt.xlabel(column1_name)
plt.ylabel(column2_name)
plt.legend([0,1,2,3])
    ##plt.legend(loc='lower right')
    
#%%
#%%
##fig, ax = plt.subplots(figsize=(20, 15))
fig, ax = plt.subplots()
ax.scatter(df[column1_name], df[column2_name], c=y_kmeans5, cmap='rainbow')
ax.set_xlabel(column1_name)
ax.set_ylabel(column2_name)
##ax.legend()
ax.legend(['cluster1', 'cluster2', 'cluster3', 'cluster4'])
    ##plt.legend(loc='lower right')
plt.show()


#%%
##fig, ax = plt.subplots(figsize=(20, 15))
fig, ax = plt.subplots()
##sns.set_palette('coolwarm')
##flatui = ["blue", "green", "red", "orange"]
flatui = sns.color_palette("pastel", cluster_number)
##sns.set_palette(flatui, n_colors = 4)
b = sns.scatterplot(ax = ax, x = column1_name, y = column2_name, data=df,
hue=df['cluster_num'], palette=flatui)

ax.set_xlabel(column1_name)
ax.set_ylabel(column2_name)
##ax.legend()
##ax.legend(['cluster1', 'cluster2', 'cluster3', 'cluster4'])
    ##plt.legend(loc='lower right')
plt.show()


#%%

fig, ax = plt.subplots()
flatui = sns.color_palette("pastel", cluster_number)
b = sns.scatterplot(ax = ax, x = column1_name, y = column2_name, data=df,
hue=df['cluster_num'], palette=flatui)

ax.set_xlabel(column1_name)
ax.set_ylabel(column2_name)
plt.show()

#%%

fig.savefig('static/uploads/kmeans.png')    
#%%
    
    
plt.savefig(img, format='png')
img.seek(0)
plot_url = base64.b64encode(img.getvalue()).decode()
aaa = plot_url


#%%
basechart = alt.Chart(df).mark_point().encode(
x= column1_name,y= column2_name, color = alt.Color('cluster_num', scale=alt.
Scale(scheme = 'dark2'))
)


basechart.save('altair.html')
