# k means aims to partation into k clusters
# k means is an unsupervised learning algorithm
# k means is a clustering algorithm
# k means is a centroid based algorithm

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#generating data
x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60,random_state=0)

#Training
model = KMeans(n_clusters=4)
y_pred= model.fit_predict(x)

#plotting result

plt.scatter(x[:,0], x[:,1],c=y_pred,cmap='rainbow')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=300, c='black',marker='X')
plt.show()