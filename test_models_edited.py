#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:15:14 2019

@author: josefigueroa
"""

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
    #svm_predictions_train = SVM.predict(X_train)
    svm_predictions_test = SVM.predict(X_test)
    print (SVM.score(X_test,Y_test))
    cm_svm = confusion_matrix(y_true=Y_test, y_pred=svm_predictions_test)
    print (cm_svm)
    print (accuracy(cm_svm))
    print ('f1_score: ',f1_score(y_true=Y_test, y_pred=svm_predictions_test, average='macro')  )
    
    test_filename = "amr-bank-struct-v1.6-test.csv"
    test_df = encode(test_filename)
    X1 = test_df.loc[:,'index':'+2POS'].values
    Y1 = test_df.loc[:,'isfocus'].values
    predictions = SVM.predict(X1)
    print (SVM.score(X1,Y1))
    cm_svm = confusion_matrix(y_true=Y1, y_pred=predictions)
    print (cm_svm)
    print (accuracy(cm_svm))
    print ('f1_score: ',f1_score(y_true=Y1, y_pred=predictions, average='macro')  )
    
    
    
    # Logistical Regression (NO)
    lr = LR()
    lr.fit(X_train, Y_train)
    correct = 0
    total = 0
    for focus, sentence in df1.groupby('focus'):
        sentence_vec = sentence.loc[:,'index':'+2POS']
        vals = np.array(lr.predict_proba(sentence_vec).tolist())
        index = np.argmin(vals[:,0])
        if (sentence.values[index][0] == 1):
            correct+=1
        total += 1
    print(correct/total)
    
    # LDA
    lda = LDA()
    lda.fit_transform(X_train, Y_train)
    lda_predictions = lda.predict(X_test)
    print(lda.score(X_test, Y_test))
    cm_lda = confusion_matrix(y_true=Y_test, y_pred=lda_predictions)
    print(cm_lda)
    
            
if __name__=="__main__":
    main()



""" Both methods fail to predict focus values, will attempt to visualize """
# Visualize mapping of each word, too much noise
"""
pca2 = PCA(n_components=2)
result = pca2.fit_transform(X_train[:1000])
color=['red','blue']
Y_train = np.array(Y_train).astype(int)

for i in range(len(result)):
    plt.scatter(result[i,0], result[i,1], color=color[Y_train[i]])
    
pc_x = pca2.explained_variance_ratio_
plt.xlabel = "PC1, {} % of variance explained".format(pc_x[0])
plt.ylabel = "PC2, {} % of variance explained".format(pc_x[1])

"""
