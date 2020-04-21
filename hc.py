# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 01:41:45 2020

@author: leno
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s = 100, c = 'red', label = 'Carefull')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],s = 100, c = 'blue', label = 'standard')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],s = 100, c = 'green', label = 'target')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],s = 100, c = 'magenta', label = 'sensible')
plt.title('clusters of clients')
plt.xlabel('annual income(k$)')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()

