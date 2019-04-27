#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:21:20 2019

@author: XiaoyiLiu
"""

#%%
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import f1_score,recall_score,precision_score
#%%
def accuracy(cm):
    return np.trace(cm)/np.sum(cm)

def svm_driver(X_train, Y_train, X_test, Y_test):
    SVM = svm.SVC(C = 1.0, kernel = 'rbf', random_state = 0)
    SVM.fit(X_train, Y_train)
    svm_predictions_train = SVM.predict(X_train)
    svm_predictions_test = SVM.predict(X_test)
    
    print ('-------------training data-----------')
    print (confusion_matrix(y_true=Y_train, y_pred=svm_predictions_train))
    print ('accuracy:',SVM.score(X_train,Y_train))
    print ('f1_score: ', f1_score(y_true=Y_train, y_pred=svm_predictions_train, average='macro'))
    print ('precision: ', precision_score(y_true=Y_train, y_pred=svm_predictions_train, average='macro')   )
    print ('recall: ', recall_score(y_true=Y_train, y_pred=svm_predictions_train, average='macro'))
    
    print ('-------------test data-------------')
    cm_svm = confusion_matrix(y_true=Y_test, y_pred=svm_predictions_test)
    print (cm_svm)
    print ('accuracy: ',SVM.score(X_test,Y_test))
    print ('f1_score: ',f1_score(y_true=Y_test, y_pred=svm_predictions_test, average='macro')  )
    print ('precision: ',precision_score(y_true=Y_test, y_pred=svm_predictions_test, average='macro')   )
    print ('recall: ', recall_score(y_true=Y_test, y_pred=svm_predictions_test, average='macro'))
    
def split(X,y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.25,random_state=1234)
    
    return X_train, X_test, Y_train, Y_test
#%%
#df1 = pd.read_csv('data/amr-bank-struct-v1.6-training.csv')
df1 = pd.read_csv('data/amr-bank-struct-v1.6-training.csv')
df1.loc[:,'POS':'+2POS'].apply(str)
df1.loc[:,'POS':'+2POS'] = df1.loc[:,'POS':'+2POS'].apply(LabelEncoder().fit_transform)
#%%
df_test = pd.read_csv('data/amr-bank-struct-v1.6-test.csv')
df_test.loc[:,'POS':'+2POS'].apply(str)
df_test.loc[:,'POS':'+2POS'] = df1.loc[:,'POS':'+2POS'].apply(LabelEncoder().fit_transform)
XX = df_test.loc[:,'index':'+2POS'].values
yy = df_test.loc[:,'isfocus'].values
#%%
print ('-------------normal-------------')
X1 = df1.loc[:,'index':'+2POS'].values
y1 = df1.loc[:,'isfocus'].values

X_train, X_test, Y_train, Y_test = split(X1,y1)
#svm_driver(X_train, Y_train, X_test, Y_test)
svm_driver(X_train, Y_train, XX, yy)
#%%
# =============================================================================
 """ down sampling the majority class """
 df_0 = df1[df1.isfocus == 0]
 df_1 = df1[df1.isfocus == 1]
 
 df_majority_downsampled = resample(df_0, 
                                  replace=False,    # sample without replacement
                                  n_samples=df_1.size,     # to match minority class
                                  random_state=123) 
 
 df_minority_upsampled = resample(df_1, 
                                  replace=True,     # sample with replacement
                                  n_samples=5000,    # to match majority class
                                  random_state=123) # reproducible results
 df_upsampled = pd.concat([df_0, df_minority_upsampled])
 df_downsampled = pd.concat([df_majority_downsampled, df_1])
 
 
 
 #%%
 print ('-------------downsampled-------------')
 X2 = df_downsampled.loc[:,'index':'+2POS'].values
 y2 = df_downsampled.loc[:,'isfocus'].values
 
 X_train, X_test, Y_train, Y_test = split(X2,y2)
 #svm_driver(X_train, Y_train, X_test, Y_test)
 svm_driver(X_train, Y_train, XX, yy)
 #%%
 print ('-------------upampled-------------')
 X3 = df_upsampled.loc[:,'index':'+2POS'].values
 y3 = df_upsampled.loc[:,'isfocus'].values
 
 X_train, X_test, Y_train, Y_test = split(X3,y3)
 #svm_driver(X_train, Y_train, X_test, Y_test)
 svm_driver(X_train, Y_train, XX, yy)
# =============================================================================
#%%
from nltk.data import find
from gensim.models import Word2Vec
import gensim.models
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
#%%
def add_embedding(df1):
    wv = []
    nor_word = list(df1['normalized_words'])
    for i in nor_word:
        if i in model:
            tmp = model[i]
    #        wv.append(np.mean(tmp))
            wv.append(tmp)
        else:
            wv.append(np.zeros(300))

    for i in range(0,300):        
        tmp = [item[i] for item in wv]
        name = 'wv'+str(i)
        df1[name] = tmp
#%%
add_embedding(df1)
add_embedding(df_test)
#%%
# =============================================================================
 print ('-------------embedding-------------')
 X4 = df1.loc[:,'index':'wv299'].values
 y4 = df1.loc[:,'isfocus'].values
 
 XX = df_test.loc[:,'index':'wv299'].values
 yy = df_test.loc[:,'isfocus'].values
 X_train, X_test, Y_train, Y_test = split(X4,y4)
 #svm_driver(X_train, Y_train, X_test, Y_test) 
 svm_driver(X_train, Y_train, XX, yy) 
# =============================================================================
#%%
LR = LogisticRegression(random_state = 1234)
X5 = df1.loc[:,'index':'wv299'].values
y5 = df1.loc[:,'isfocus'].values
X_train, X_test, Y_train, Y_test = split(X5,y5)

XX = df_test.loc[:,'index':'wv299'].values
yy = df_test.loc[:,'isfocus'].values
LR.fit(X5, y5)
#%%
y_pred_t = LR.predict(X5)
y_pred = LR.predict(XX)
print ('Logistic Regression')
print ('-------------training data-----------')
print (confusion_matrix(y_true=y5, y_pred=y_pred_t))
print ('accuracy:',LR.score(X5,y5))
print ('f1_score: ', f1_score(y_true=y5, y_pred=y_pred_t, average='macro'))
print ('precision: ', precision_score(y_true=y5, y_pred=y_pred_t, average='macro')   )
print ('recall: ', recall_score(y_true=y5, y_pred=y_pred_t, average='macro'))

print ('-------------test data-------------')
cm_svm = confusion_matrix(y_true=yy, y_pred=y_pred)
print (cm_svm)
print ('accuracy: ',LR.score(XX,yy))
print ('f1_score: ',f1_score(y_true=yy, y_pred=y_pred, average='macro'))
print ('precision: ',precision_score(y_true=yy, y_pred=y_pred, average='macro'))
print ('recall: ', recall_score(y_true=yy, y_pred=y_pred, average='macro'))
#%%

df_test['prediction'] = y_pred
for i in range(0,300):
    name = 'wv'+ str(i)
    df_test = df_test.drop([name],axis=1)


#%%   
fname = "lr_out_little_prince.csv"
df_test.to_csv(fname,index = None, header=True)









