# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 23:59:40 2021

@author: Kedar
"""
###Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("D:\Datasets_PCA (1)\wine.csv")
df1=pd.read_csv("D:\Datasets_PCA (1)\wine.csv")

df1.columns
df1.isna().sum()
df1.drop_duplicates(inplace=True)
df1.mean()
df1.median()
df1.var()
df1.skew()
df1.kurt()

####Histograms to understand distribution of data among features
plt.hist(df1['Alcohol'])
plt.hist(df1['Malic'])  
plt.hist(df1['Ash'])
plt.hist(df1['Alcalinity']) 
plt.hist(df1['Magnesium'])  
plt.hist(df1['Phenols'])  
plt.hist(df1['Flavanoids']) 
plt.hist(df1['Nonflavanoids'])  
plt.hist(df1['Proanthocyanins'])  
plt.hist(df1['Color'])  
plt.hist(df1['Hue']) 
plt.hist(df1['Dilution']) 
plt.hist(df1['Proline']) 





###Boxplots to get an idea about outliers
plt.boxplot(df1['Alcohol']) ###No outlier
plt.boxplot(df1['Malic'])  ###Few outliers at higher end
plt.boxplot(df1['Ash'])   ###Few outliers at higher end
plt.boxplot(df1['Alcalinity'])  ###Few outliers at higher end
plt.boxplot(df1['Magnesium'])  ###Few outliers at higher end
plt.boxplot(df1['Phenols'])  ###No outliers
plt.boxplot(df1['Flavanoids']) ###No outliers
plt.boxplot(df1['Nonflavanoids'])  ###No outliers
plt.boxplot(df1['Proanthocyanins'])  ###Few outliers at higher end
plt.boxplot(df1['Color'])  ###Few outliers at higher end
plt.boxplot(df1['Hue']) ###Few outliers at higher end
plt.boxplot(df1['Dilution']) ###No outliers
plt.boxplot(df1['Proline']) #No outliers


###Winsorization 
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Malic','Ash','Alcalinity','Magnesium','Proanthocyanins','Color','Hue'])
df_o=winsor.fit_transform(df1[['Malic','Ash','Alcalinity','Magnesium','Proanthocyanins','Color','Hue']])

###Preparing datasets for normalization
df1.drop(['Malic','Ash','Alcalinity','Magnesium','Proanthocyanins','Color','Hue'],axis=1,inplace=True)
df1=pd.concat([df1,df_o],axis=1)
df_t=pd.DataFrame([[df1['Type']]])
df1.drop(['Type'],axis=1,inplace=True)


####Normalization fucntion
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)

df1=norm_func(df1)
df1.describe()

####K means clustering before PCA
TWSS = []
k = list(range(2, 9))

from sklearn.cluster import KMeans
for i in k:
    kmeans=KMeans(n_clusters=i,random_state=3425)
    kmeans.fit_transform(df1)
    TWSS.append(kmeans.inertia_)

###Scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel('Number of Clusters');plt.ylabel('total within SS')

model=KMeans(n_clusters=3)
model.fit(df1)
model.labels_
mb=pd.Series(model.labels_)
df['clust1K'] = mb ###clustered column


###Hierarchical clustering before PCA
###To create dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch


z=linkage(df1,method='complete',metric='euclidean')

###Dendrogram
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z)
plt.show()

###Agglomerative clustering
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


