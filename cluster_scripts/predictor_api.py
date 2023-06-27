"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np
from sklearn.externals import joblib
import re
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import numpy as np


# Load the models 
# model_dict is the collection of extra tree models 

# This line doesn't work, joblib only loads locally. File is too big to upload to heroku though
# model_dict = joblib.load('https://drive.google.com/open?id=1h20N5Cooti2e5CDkmKY5LOzRuLksyR5e')
# model_dict = joblib.load('./static/models/models_compressed.p')
# word_vectorizer = joblib.load('static/models/word_vectorizer.p')



def raw_chat_to_model_input(raw_input_string):
    # Converts string into cleaned text
    cleaned_text = int(raw_input_string)
    return word_vectorizer.transform(cleaned_text)


def make_prediction(input_chat):
    """

    """
    cluster_number = int(input_chat)


    iris = datasets.load_iris()

    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    x = df.iloc[:, [0, 1, 2, 3]].values
    kmeans5 = KMeans(n_clusters=cluster_number)
    y_kmeans5 = kmeans5.fit_predict(x)
    column_list = df.columns
    img = BytesIO()
    plt.scatter(x[:, 0], x[:, 3], c=y_kmeans5, cmap='rainbow')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[3])
    ##plt.legend(loc='lower right')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    aaa = plot_url
    return (input_chat, aaa)

# This section checks that the prediction code runs properly
# To test, use "python predictor_api.py" in the terminal.

# if __name__='__main__' section only runs
# when running this file; it doesn't run when importing

if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what empty string predicts")
    print('input string is ')
    chat_in = '3'
    pprint(chat_in)

    x_input, probs = make_prediction(chat_in)
    print(f'Input values: {x_input}')
    print('Output probabilities')
    pprint(probs)
