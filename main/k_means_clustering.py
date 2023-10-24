import numpy as np


class KMeansClustering:

    def __init__(self, train_data, num_clusters=2, max_iter=100, threshold=1e-3):
        """
        Initializes a KMeansClustering object.

        :param train_data: The training data.
        :param num_clusters: The number of clusters to create.
        :param max_iter: The maximum number of iterations for the K-Means algorithm.
        :param threshold: The convergence threshold for centroid updates.
        """

        self.train_data = train_data
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.num_dims = len(self.train_data[0])
        self.threshold = threshold

        self.clusters = {}  # a dictionary to store the detected clusters

    def train(self):
        """
        Train the KMeans model on the provided data.
        """

        data = np.array(self.train_data)
        prev_centroids = self.initialization(data, self.num_clusters)
        cluster = [0 for i in range(len(self.train_data))]
        cluster_keys = [i for i in range(self.num_clusters)]
        change_centroids = 1e+10

        for i in range(self.max_iter):

            if change_centroids > self.threshold:

                cluster = self.cluster(self.train_data, self.num_clusters, prev_centroids)

                new_centroids = self.centroids(self.train_data, self.num_clusters, cluster)
                change_centroids = self.measure_change(new_centroids, prev_centroids)
                prev_centroids = new_centroids

        for key in cluster_keys:
            for j in range(len(self.train_data)):
                if key == cluster[j]:
                    if f"Cluster {key+1}" not in self.clusters:
                        self.clusters[f"Cluster {key+1}"] = [self.train_data[j]]
                    else:
                        self.clusters[f"Cluster {key+1}"].append(self.train_data[j])

    def initialization(self, data, num_clusters):
        """
        Initialize cluster centroids using random data points from the dataset.

        :param data: The dataset for initialization.
        :param num_clusters: The number of clusters to create.
        :return: A NumPy array of initial cluster centroids.
        """

        centroids = []
        num_features = len(data[0])

        for i in range(num_clusters):
            centroid = []
            for j in range(num_features):
                cx = np.random.uniform(min(data[:, j]), max(data[:, j]))
                centroid.append(cx)

            centroids.append(centroid)

        return np.asarray(centroids)

    def cluster(self,data, num_cluster, prev_centroids):
        """
        Assign data points to clusters based on the closest centroid.

        :param data: The dataset to cluster.
        :param num_cluster: The number of clusters.
        :param prev_centroids: The previous cluster centroids.
        :return: An array of cluster assignments.
        """

        cluster = [-1 for _ in range(len(data))]
        for i in range(len(data)):
            dist_arr = []
            for j in range(num_cluster):
                dist_arr.append(self.distance(data[i], prev_centroids[j]))
            idx = np.argmin(dist_arr)
            cluster[i] = idx
        return np.asarray(cluster)

    def distance(self, a, b):
        """
        Compute the Euclidean distance between two data points.

        :param a: The first data point.
        :param b: The second data point.
        :return: The Euclidean distance between a and b.
        """
        distance = np.sqrt(sum(np.square(a - b)))
        return distance

    def centroids(self, data, num_cluster, cluster):
        """
        Compute new centroids for each cluster based on the assigned data points.

        :param data: The dataset.
        :param num_cluster: The number of clusters.
        :param cluster: The cluster assignments for each data point.
        :return: An array of updated cluster centroids.
        """
        cg_arr = []
        for i in range(num_cluster):
            arr = []
            for j in range(len(data)):
                if cluster[j] == i:
                    arr.append(data[j])
            cg_arr.append(np.mean(arr, axis=0))
        return np.asarray(cg_arr)

    def measure_change(self, prev_centroids, new_centroids):
        """
        Measure the change in centroids between iterations to check for convergence.

        :param prev_centroids: The centroids from the previous iteration.
        :param new_centroids: The updated centroids in the current iteration.
        :return: The measure of change in centroids.
        """
        res = 0
        for a, b in zip(prev_centroids, new_centroids):
            res += self.distance(a, b)
        return res

