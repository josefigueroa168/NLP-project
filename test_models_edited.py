#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:15:14 2019

@author: josefigueroa
"""

#%%
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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import f1_score
import sys
#%%

def tag2vec(data_df, tag_map):
    trim_data = data_df.loc[:,'POS':'+2POS']
    trim_data['POS'] = trim_data['POS'].apply(lambda x: tag_map[x])
    trim_data['-1POS'] = trim_data['-1POS'].apply(lambda x: tag_map[x])
    trim_data['-2POS'] = trim_data['-2POS'].apply(lambda x: tag_map[x])
    trim_data['+1POS'] = trim_data['+1POS'].apply(lambda x: tag_map[x])
    trim_data['+2POS'] = trim_data['+2POS'].apply(lambda x: tag_map[x])
    return trim_data

def accuracy(cm):
    return np.trace(cm)/np.sum(cm)

def encode(filename):
    df1 = pd.read_csv(filename)
    df1.loc[:,'POS':'+2POS'].apply(str)
    df1.loc[:,'POS':'+2POS'] = df1.loc[:,'POS':'+2POS'].apply(LabelEncoder().fit_transform)
    return df1

def downsample(df_0, df_1):
    df_majority_downsampled = resample(df_0, 
                                 replace=False,    # sample without replacement
                                 n_samples=df_1.size,     # to match minority class
                                 random_state=123) 
    return pd.concat([df_majority_downsampled, df_1])


def main():
    #filename = sys.arg[1]
    filename = "amr-bank-struct-v1.6-training.csv"
    df1 = encode(filename)
    df_downsampled = downsample(df1[df1.isfocus == 0], df1[df1.isfocus == 1])
    X1 = df_downsampled.loc[:,'index':'+2POS'].values
    y1 = df_downsampled.loc[:,'isfocus'].values
    X_train, X_test, Y_train, Y_test = train_test_split(
        X1, y1, test_size=0.25, stratify = y1,random_state=300)
    
    # SVM
    SVM = svm.SVC(C = 1.0, kernel = 'rbf', random_state = 0)
    SVM.fit(X_train, Y_train)
    svm_predictions_train = SVM.predict(X_train)
    svm_predictions_test = SVM.predict(X_test)
    print (SVM.score(X_test,Y_test))
    cm_svm = confusion_matrix(y_true=Y_test, y_pred=svm_predictions_test)
    print (cm_svm)
    print (accuracy(cm_svm))
    print ('f1_score: ',f1_score(y_true=Y_test, y_pred=svm_predictions_test, average='macro')  )

if __name__=="__main__":
    main()
#%%

data_df = pd.read_csv("data/amr-bank-struct-v1.6-training.csv")
data_df = data_df.applymap(str)
labels = data_df.loc[:,'isfocus']
sentences = data_df.loc[:,'POS':'+2POS'].values.tolist()
model = Word2Vec(sentences)
words = list(model.wv.vocab)
X = model[model.wv.vocab]

#%%
""" trying another encode way """
df1 = pd.read_csv('data/amr-bank-struct-v1.6-training.csv')
df1.loc[:,'POS':'+2POS'].apply(str)
df1.loc[:,'POS':'+2POS'] = df1.loc[:,'POS':'+2POS'].apply(LabelEncoder().fit_transform)

#%%
""" down sampling the majority class """
df_0 = df1[df1.isfocus == 0]
df_1 = df1[df1.isfocus == 1]

df_majority_downsampled = resample(df_0, 
                                 replace=False,    # sample without replacement
                                 n_samples=df_1.size,     # to match minority class
                                 random_state=123) 

df_downsampled = pd.concat([df_majority_downsampled, df_1])


#%%
X1 = df_downsampled.loc[:,'index':'+2POS'].values
y1 = df_downsampled.loc[:,'isfocus'].values

#%%

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

#%%

#Split to test and train
# =============================================================================
# X_train, X_test, Y_train, Y_test = train_test_split(
#         trim_data, labels, test_size=0.2, random_state=300)
# =============================================================================

X_train, X_test, Y_train, Y_test = train_test_split(
        X1, y1, test_size=0.25, stratify = y1,random_state=300)

#%%
""" Linear Discrimination Analysis """
lda = LDA()
lda.fit_transform(X_train, Y_train)
predictions = lda.predict(X_test)
#predictions = lda.predict(X_train)
# These scores dont account for the importance of actually tagging focus
lda.score(X_train, Y_train)
lda.score(X_test, Y_test)

# Shows 0%
cm = confusion_matrix(y_true=Y_test, y_pred=predictions)
print (cm)
print (accuracy(cm))
    
""" Linear Discrimination Analysis """
#%%
""" SVM """
#SVM = svm.SVC(gamma='scale')
SVM = svm.SVC(C = 1.0, kernel = 'rbf', random_state = 0)
SVM.fit(X_train, Y_train)
svm_predictions_train = SVM.predict(X_train)
svm_predictions_test = SVM.predict(X_test)
print (SVM.score(X_test,Y_test))
cm_svm = confusion_matrix(y_true=Y_test, y_pred=svm_predictions_test)
print (cm_svm)
print (accuracy(cm_svm))
print ('f1_score: ',f1_score(y_true=Y_test, y_pred=svm_predictions_test, average='macro')  )

""" SVM """
#%%
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
    
#%% 

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


