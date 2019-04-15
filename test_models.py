#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:15:14 2019

@author: josefigueroa
"""

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm
import matplotlib.pyplot as plt



data_df = pd.read_csv("amr-bank-struct-v1.6-training.csv")
data_df = data_df.applymap(str)
labels = data_df.loc[:,'isfocus']
sentences = data_df.loc[:,'POS':'+2POS'].values.tolist()
model = Word2Vec(sentences)
words = list(model.wv.vocab)


X = model[model.wv.vocab]

pca = PCA(n_components=1)
result = pca.fit_transform(X)
	
#plt.scatter(result[:, 0], result[:, 1])

#for i, word in enumerate(words):
#	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

tag_map = {}
for i in range(len(words)):
    tag_map[words[i]] = result[i]
    
data_df = data_df.loc[:,'POS':'+2POS']
data_df['POS'] = data_df['POS'].apply(lambda x: tag_map[x])
data_df['-1POS'] = data_df['-1POS'].apply(lambda x: tag_map[x])
data_df['-2POS'] = data_df['-2POS'].apply(lambda x: tag_map[x])
data_df['+1POS'] = data_df['+1POS'].apply(lambda x: tag_map[x])
data_df['+2POS'] = data_df['+2POS'].apply(lambda x: tag_map[x])

X_train, X_test, Y_train, Y_test = train_test_split(
        data_df, labels, test_size=0.2, random_state=300)

""" Linear Discrimination Analysis """
lda = LDA()
lda.fit_transform(X_train, Y_train)
predictions = lda.predict(X_test)
predictions = lda.predict(X_train)
lda.score(X_train, Y_train)
lda.score(X_test, Y_test)
confusion_matrix(y_true=Y_test, y_pred=predictions)
    
""" Linear Discrimination Analysis """

""" SVM """
SVM = svm.SVC(gamma='scale')
SVM.fit(X_train, Y_train)
svm_predictions = SVM.predict(X_train)
confusion_matrix(y_true=Y_train, y_pred=svm_predictions)

""" SVM """

""" Both methods fail to predict focus values, will attempt to visualize """
pca2 = PCA(n_components=2)
result = pca2.fit_transform(X_train)
color=['red','blue']
Y_train = np.array(Y_train).astype(int)

for i in range(len(result)):
    plt.scatter(result[i,0], result[i,1], color=color[Y_train[i]])
    
pc_x = pca2.explained_variance_ratio_
plt.xlabel = "PC1, {} % of variance explained".format(pc_x[0])
plt.ylabel = "PC2, {} % of variance explained".format(pc_x[1])

# Too much noise

