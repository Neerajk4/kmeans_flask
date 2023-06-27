import os
from os import path
from app import app
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pandas as pd
from cluster_functions import kmeans_run, kmeans_plot, cluster_main
from k_means_inter import kmeans_iter, initialize_centroids, assign_clusters, reclassify_centroids, optimize_check

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/iris_test',methods = ['GET', 'POST'])
def iris_demo():
    if request.method == 'POST':
        column1_name = request.form['column1_name']
        column2_name = request.form['column2_name']
        n_clusters = request.form['n_clusters']
        print(column1_name)
        print(column2_name)
        print(n_clusters)
        print(type(column1_name))
        print(type(column2_name))
        print(type(n_clusters))
        n_clusters = int(n_clusters)
        tol = 0.0001
        max_iter = 6
        cluster_main(column1_name, column2_name, n_clusters)
        column_names = [column1_name, column2_name]
        kmeans_iter(n_clusters, tol, max_iter, column_names)
        file_exist = path.exists("static/uploads/kmeans.png")
        return render_template('iris_test.html', file_exist = file_exist)
    else:
        file_exist = path.exists("static/uploads/kmeans.png")
        return render_template('iris_test.html', file_exist=file_exist)


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='static/uploads/' + filename), code=301)

@app.route('/iris_iteration')
def iter_demo():
    file_exist = path.exists("static/uploads/kmeans_iteration.png")
    return render_template('iris_iter.html', file_exist=file_exist)

if __name__ == "__main__":
    app.run()