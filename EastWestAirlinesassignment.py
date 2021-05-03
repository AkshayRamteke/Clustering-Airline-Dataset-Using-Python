

import pandas as pd
import matplotlib.pylab as plt
flight=pd.read_csv("E:/Assignment/7_Clustering/EastWestAirlines.csv")

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
    
df_norm= norm_func(flight.iloc[:,1:])
df_norm
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method='complete',metric='euclidean')

plt.figure(figsize=(15,5));plt.title("hierarchial clustering dendogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=6)

from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster as skc

h_clust=AgglomerativeClustering(n_clusters=14,affinity="euclidean",linkage="complete").fit(df_norm)
h_clust

h=pd.Series(h_clust.labels_)
flight['clust']=h
flight1=flight.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

flight2= flight1.iloc[:,2:].groupby(flight1.clust).median()

