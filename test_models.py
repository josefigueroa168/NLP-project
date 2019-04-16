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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def tag2vec(data_df, tag_map):
    trim_data = data_df.loc[:,'POS':'+2POS']
    trim_data['POS'] = trim_data['POS'].apply(lambda x: tag_map[x])
    trim_data['-1POS'] = trim_data['-1POS'].apply(lambda x: tag_map[x])
    trim_data['-2POS'] = trim_data['-2POS'].apply(lambda x: tag_map[x])
    trim_data['+1POS'] = trim_data['+1POS'].apply(lambda x: tag_map[x])
    trim_data['+2POS'] = trim_data['+2POS'].apply(lambda x: tag_map[x])
    return trim_data


data_df = pd.read_csv("amr-bank-struct-v1.6-training.csv")
data_df = data_df.applymap(str)
labels = data_df.loc[:,'isfocus']
sentences = data_df.loc[:,'POS':'+2POS'].values.tolist()
model = Word2Vec(sentences)
words = list(model.wv.vocab)
X = model[model.wv.vocab]

#Dimensionality Reduction
pca = PCA(n_components=1)
result = pca.fit_transform(X)

# For Dimensions = 2
#plt.scatter(result[:, 0], result[:, 1])
#for i, word in enumerate(words):
#	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

#Correlate Each tag with a numerical vaue
tag_map = {}
for i in range(len(words)):
    tag_map[words[i]] = result[i]
    
#Create a 5 dimension array correlating with each word
trim_data = (data_df, tag_map)

#Split to test and train
X_train, X_test, Y_train, Y_test = train_test_split(
        trim_data, labels, test_size=0.2, random_state=300)

""" Linear Discrimination Analysis """
lda = LDA()
lda.fit_transform(X_train, Y_train)
predictions = lda.predict(X_test)
predictions = lda.predict(X_train)
# These scores dont account for the importance of actually tagging focus
lda.score(X_train, Y_train)
lda.score(X_test, Y_test)

# Shows 0%
confusion_matrix(y_true=Y_test, y_pred=predictions)
    
""" Linear Discrimination Analysis """

""" SVM """
SVM = svm.SVC(gamma='scale')
SVM.fit(X_train, Y_train)
svm_predictions = SVM.predict(X_train)
confusion_matrix(y_true=Y_train, y_pred=svm_predictions)

""" SVM """

""" Logistical Regression """
lr = LR(class_weight={'1':2})
lr.fit(X_train, Y_train)
numeric_train = X_train.values
predict = lr.predict_proba(numeric_train).tolist()
correct = 0
total = 0
for focus, sentence in data_df.groupby('focus'):
    sentence_vec = tag2vec(sentence, tag_map)
    vals = np.array(lr.predict_proba(sentence_vec).tolist())
    index = np.argmin(vals[:,0])
    print(index)
    if (sentence.values[index][2] == '1'):
        correct+=1
    total += 1
print(correct/total)

"""Logistical Regression"""
    
    

""" Both methods fail to predict focus values, will attempt to visualize """
# Visualize mapping of each word, too much noise
pca2 = PCA(n_components=2)
result = pca2.fit_transform(X_train)
color=['red','blue']
Y_train = np.array(Y_train).astype(int)

for i in range(len(result)):
    plt.scatter(result[i,0], result[i,1], color=color[Y_train[i]])
    
pc_x = pca2.explained_variance_ratio_
plt.xlabel = "PC1, {} % of variance explained".format(pc_x[0])
plt.ylabel = "PC2, {} % of variance explained".format(pc_x[1])


