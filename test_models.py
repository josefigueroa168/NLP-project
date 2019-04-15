#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:15:14 2019

@author: josefigueroa
"""

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data_df = pd.read_csv("amr-bank-struct-v1.6-training.csv")
data_df = data_df.applymap(str)
sentences = data_df.loc[:,'POS':'+2POS'].values.tolist()
model = Word2Vec(sentences)
words = list(model.wv.vocab)


X = model[model.wv.vocab]

pca = PCA(n_components=1)
result = pca.fit_transform(X)
	
plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

tag_map = {}
for i in range(len(words)):
    tag_map[words[i]] = result[i]
    
numerical_data = pd.DataFrame()