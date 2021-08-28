import pandas as pd
import numpy as np
df=pd.read_csv('/content/drive/MyDrive/wine.csv')
df1=df.copy()
df1=df1.drop(['Type'],axis=1)
# Normalization
from sklearn.preprocessing import MinMaxScaler
MS=MinMaxScaler()
df1=pd.DataFrame(MS.fit_transform(df1),columns=df1.columns)

# PCA
from sklearn.decomposition import PCA
p=PCA(n_components=13)
fit=p.fit_transform(df1)
var=p.explained_variance_ratio_
cumvar=np.cumsum(np.round(var,decimals=4)*100)
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,6)
plt.ylim(0.0,1.1)
fig, ax = plt.subplots()
x=np.arange(0,13,step=1)
y = np.cumsum(p.explained_variance_ratio_)
plt.plot(x, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 13, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(0.5, 0.85, '90% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

data=pd.DataFrame(fit)
#len(df.columns)
pca_final=pd.concat([df.Type,data.iloc[:,0:6]],axis=1)
pca_final
