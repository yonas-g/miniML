# miniML

```Python

from miniML.clustering import KMeans

kmeans = KMeans(n_clusters = 3)

kmeans.fit(X)

# predict to which cluster each value belongs
predictions = kmeans.predict(X)

# n_cluster center points
centers = kmeans.cluster_centers_

```
