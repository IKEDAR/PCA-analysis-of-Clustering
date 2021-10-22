# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:26:02 2021

@author: Kedar
"""
###Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###Importing datasets
df=pd.read_csv('D:\Datasets_PCA (1)\heart disease.csv')
df1=pd.read_csv('D:\Datasets_PCA (1)\heart disease.csv')

###Description, null value and moments of business decision
df1.describe()
df1.isnull().sum()
df1.mean()
df1.median()
df1.var()
df1.std()
df1.skew()
df1.kurt()

###Boxplots to get an idea about outliers
plt.boxplot(df1['age'])      ###No outliers
plt.boxplot(df1['trestbps'])   ###Outliers at higher end
plt.boxplot(df1['chol'])   ###Outliers at higher end
plt.boxplot(df1['thalach'])  ###One outlier value at lower end
plt.boxplot(df1['oldpeak'])  ###Outliers at higher end

###Histograms to understand spread of data
plt.hist(df1['age'])
plt.hist(df1['trestbps'])
plt.hist(df1['chol'])
plt.hist(df1['thalach'])
plt.hist(df1['oldpeak'])

###Winsorizer for outlier treatment
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=(['age','trestbps','chol','thalach','oldpeak']))
df_o=winsor.fit_transform(df1[['age','trestbps','chol','thalach','oldpeak']])

####Boxplots to make sure there are no outliers present
plt.boxplot(df_o['age'])      
plt.boxplot(df_o['trestbps'])   
plt.boxplot(df_o['chol'])   
plt.boxplot(df_o['thalach'])  
plt.boxplot(df_o['oldpeak'])

df1.drop(['age','trestbps','chol','thalach','oldpeak'],axis=1,inplace=True)

###Normalization function
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

df_n=norm_func(df_o)
df_n.describe()

df1=pd.concat([df1,df_n],axis=1)




####K means clustering before PCA
TWSS = []
k = list(range(2, 9))

from sklearn.cluster import KMeans
for i in k:
    kmeans=KMeans(n_clusters=i,random_state=3425)
    kmeans.fit_transform(df1)
    TWSS.append(kmeans.inertia_)
    
plt.plot(k,TWSS,'ro-');plt.xlabel('Number of Clusters');plt.ylabel('total within SS')

model=KMeans(n_clusters=3)
model.fit(df1)
model.labels_
mb=pd.Series(model.labels_)
df['clust1K'] = mb




###Hierarchical clustering before PCA
###To create dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch


z=linkage(df1,method='complete',metric='euclidean')

###Dendrogram
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z)
plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df1) 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df['clust1H']=cluster_labels



####Performing PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca_values=pca.fit_transform(df1)


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var


# PCA weights
pca.components_
pca.components_[0]


# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

pca_values
df_pca = pd.DataFrame(pca_values)


####K means clustering after PCA
TWSS = []
k = list(range(2, 9))

from sklearn.cluster import KMeans
for i in k:
    kmeans=KMeans(n_clusters=i,random_state=3425)
    kmeans.fit_transform(df_pca)
    TWSS.append(kmeans.inertia_)
    
plt.plot(k,TWSS,'ro-');plt.xlabel('Number of Clusters');plt.ylabel('total within SS')

model=KMeans(n_clusters=3)
model.fit(df_pca)
model.labels_
mb=pd.Series(model.labels_)
df['clust2K'] = mb



###Hierarchical clustering after PCA
###To create dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch


z=linkage(df_pca,method='complete',metric='euclidean')

###Dendrogram
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z)
plt.show()

###Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_pca) 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df['clust2H']=cluster_labels

###grouping datasets by clusters
df.groupby(df.clust1H).head()
df.groupby(df.clust2H).head()






