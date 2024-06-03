import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_origin = pd.read_csv('ckd-dataset-v2.csv') # Dữ liệu tập huấn
full_data_scanning = pd.read_csv('ckd-dataset-v2.csv')
data_origin = data_origin.replace("Dec-20", "12 - 20")
n = None    # Độ sâu của cây quyết định
if 'affected' in data_origin.columns:
    data_origin = data_origin.drop(columns='affected') 
    data_origin = data_origin.drop(columns='class ckd') 

x = data_origin
# y = data_origin.loc[:, 'class ckd']

dic = {}
for i in range(0, len(x.columns)):
    dic.update({x.columns[i]: x.iloc[:,i].unique()})

Xnorm = x.copy()

for i in range(0, len(x.columns)):
    if x.iloc[:,i].dtype == 'object':
        Xnorm.iloc[:,i] = x.iloc[:,i].astype('category').cat.codes

Xnorm = StandardScaler().fit_transform(Xnorm)

kmeans = KMeans(n_clusters=2, random_state=0).fit(Xnorm)
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(Xnorm)
print(pred_label)


# KMedoids
kmedoids = KMedoids(n_clusters=2, random_state=0).fit(Xnorm)
print(kmedoids.cluster_centers_)
pred_label_m = kmedoids.predict(Xnorm)
print(pred_label_m)
