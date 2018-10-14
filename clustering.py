import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

x= np.array([[1,2],[1.4,3],[3,1.1],[7,8],[4,6],[8,4]])
#plt.scatter(x[:,0],x[:,1],s=150)
#plt.show()
clf=KMeans(n_clusters=2)
clf.fit(x)
centroids=clf.cluster_centers_
labels=clf.labels_

colors=["g.","r.","c.","b.","k.","o."]
for i in range(len(x)):
    plt.plot(x[i][0],x[i][1],colors[labels[i]],markersize= 10)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=5)
plt.show()
