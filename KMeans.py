from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x,y=make_blobs(n_samples=300,centers=4,cluster_std=0.5,random_state=0)

# print(x[:,1])

model=KMeans(n_clusters=4)
model.fit(x)
centers=model.cluster_centers_

print(centers)
plt.scatter(x[:,0],x[:,1])
plt.scatter(centers[0,0],centers[0,1],c='blue',label="Centroid 1")
plt.legend(frameon=True)
plt.show()