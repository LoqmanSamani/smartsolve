class KMeansClustering:

    """
    the training_data & validation_data must both have this structure:
    list = [[x11,x12,...,x1n], [x21,x22,x2n], [xm1,xm2,...,xmn]]

    """
    def __init__(self, training_data, validation_data=None, num_clusters=2, num_iterations=50, distance='Euclidean', initializer='KMeans++'):

        self.training_data = training_data + validation_data
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        self.distance = distance  # The available methods are Euclidean, Manhattan, Cosine, Hamming & JaccardSimilarity
        self.initializer = initializer






    def random_initialization(self):

        import numpy as np

        num_dimensions = len(self.training_data[0])

        random_dimensions = []

        for i in range(num_dimensions):

            dimension = [point[i] for point in self.training_data]
            mean = np.mean(dimension)
            std = np.std(dimension)

            random_d = []
            for j in range(self.num_clusters):

                random_num = np.random.normal(mean, std)
                random_d.append(random_num)

            random_dimensions.append(random_d)

        centroids = list(zip(*random_dimensions))

        return centroids







    def k_means_plus_initialization(self):

        import numpy as np

        num_dimensions = len(self.training_data[0])

        centroids = []

        centroid1 = []

        for i in range(num_dimensions):

            dimension = [point[i] for point in self.training_data]

            mean = np.mean(dimension)

            std = np.std(dimension)

            random_num = np.random.normal(mean, std)

            centroid1.append(random_num)

        centroids.append(centroid1)

        for i in range(self.num_clusters - 1):

             distances = self.euclidean_distance(centroids[-1])

             extended_distances = []

             for distance in distances:

                 extended_distances.extend(distance)

             probabilities = np.array(extended_distances) / np.sum(extended_distances)

             new_centroid = self.training_data[np.random.choice(self.training_data.shape[0], p=probabilities)]

             centroids.append(new_centroid)


        return centroids








    def euclidean_distance(self, centroids):
        """
        Euclidian Distance is suitable for continuous numerical data.
        The euclidian distance between two points A and B in a multidimensional
        space is the length of the straight line connecting them. In general,
        for points given by Cartesian coordinates in n n-dimensional Euclidean
        space, the distance is:

        d(p, q) = square_root((p1 - q1)² + (p2 - q2)² + ... + (pn - qn)²)
        """

        import numpy as np

        distances = []   # a list of distances for each point from each centroid.

        for point in self.training_data:

            point_distances = []

            for centroid in centroids:

                point_distance = np.sqrt(np.sum([np.power([point[i]-centroid[i]], 2) for i in range(len(point))]))

                point_distances.append(point_distance)

            distances.append(point_distances)

        return distances







    def manhattan_distance(self, centroids):

        """
        it measures the distance between two points as the sum
        of the absolute differences between their coordinates.
        It's suitable for cases where the data lies on a grid-like structure.

        d(p, q) = |p1 - q1| + |p2 - q2| + ... + |pn - qn|
        """

        import numpy as np

        distances = []

        for point in self.training_data:

            point_distances = []

            for centroid in centroids:

                point_distance = np.sum([abs(point[i]-centroid[i]) for i in range(len(point))])

                point_distances.append(point_distance)

            distances.append(point_distances)

        return distances







    def cosine_similarity(self, centroids):

        """
        Used for high-dimensional data like text or document data.
        It measures the cosine of the angle between two vectors,
        which indicates their similarity in terms of orientation.

        d(p, q) = ((p1 * q1) + (p2 * q2) + ... + (pn * qn)) / square_root((p1²+p2²+ ...+pn²)*(q1²+q2²+ ...+qn²))
        """
        import numpy as np

        distances = []

        for point in self.training_data:

            point_distances = []

            for centroid in centroids:

                point_distance = (np.sum([point[i] * centroid[i] for i in range(len(point))]) /
                                  np.sqrt(sum([np.power(i, 2) for i in point])*sum([np.power(p, 2) for p in point])))

                point_distances.append(point_distance)

            distances.append(point_distances)

        return distances






    def hamming_distance(self, centroids):

        """
        Used for categorical data or binary data. It calculates the number
        of positions at which the corresponding symbols are different.
        """

        import numpy as np

        distances = []

        for point in self.training_data:

            point_distances = []

            for centroid in centroids:

                point_distance = np.sum([1 if centroid[i] != point[i] else 0 for i in range(len(point))])

                point_distances.append(point_distance)

            distances.append(point_distances)

        return distances







    def jaccard_similarity_index(self, centroids):

        """
        Used for sets or binary data.It calculates the size of the
        intersection of two sets divided by the size of their union.
        Often used for text analysis and recommendation systems.

        similarity(p, q) = a / (a + b + c)

        a = the number of attributes that equal 1 for both objects i and j
        b = the number of attributes that equal 0 for object i but equal 1 for object j
        c = the number of attributes that equal 1 for object i but equal 0 for object j
        """

        import numpy as np

        similarity = []

        for point in self.training_data:

            point_similarities = []

            for centroid in centroids:

                a = np.sum([1 if centroid[i] == 1 and point[i] == 1 else 0 for i in range(len(point))])
                b = np.sum([1 if centroid[i] == 0 and point[i] == 1 else 0 for i in range(len(point))])
                c = np.sum([1 if centroid[i] == 1 and point[i] == 0 else 0 for i in range(len(point))])

                point_similarity = a / (a + b + c)

                point_similarities.append(point_similarity)

            similarity.append(point_similarities)

        return similarity








    def identify_clusters(self, threshold=0.01):

        import numpy as np

        clusters = {}

        cluster_keys = [f"cluster {i}" for i in range(1, self.num_clusters + 1)]

        centroids = None
        distances = None

        try:

            if self.initializer == 'KMeans++':
                centroids = self.k_means_plus_initialization()

            elif self.initializer == 'Random':
                centroids = self.random_initialization()

            else:

                raise ValueError("Invalid initializer. Choose from 'KMeans++' or 'Random'.")

        except ValueError as e:

            print(str(e))



        for i in range(self.num_iterations):

            try:

                if self.distance == 'Euclidean':
                    distances = self.euclidean_distance(centroids)

                elif self.distance == 'Manhattan':
                    distances = self.manhattan_distance(centroids)

                elif self.distance == 'Cosine':
                    distances = self.cosine_similarity(centroids)

                elif self.distance == 'Hamming':
                    distances = self.hamming_distance(centroids)

                elif self.distance == 'JaccardSimilarity':
                    distances = self.jaccard_similarity_index(centroids)

                else:
                    raise ValueError("Please select a distance calculation method (distance = ?)."
                                     "The available methods are Euclidean, Manhattan, Cosine, Hamming & JaccardSimilarity")

            except ValueError as e:

                print(str(e))



            nearest_cluster = [distance.index(min(distance)) for distance in distances]

            for k in range(len(cluster_keys)):

                # the points, which are clustered in the same cluster
                cluster_points = [self.training_data[n] for n in range(len(nearest_cluster))
                                  if cluster_keys[k] == nearest_cluster[n]]

                clusters[cluster_keys[k]] = cluster_points

            new_centroids = []  # a list for updated centroids

            for val in clusters.values():

                lst = [list(zip(*point)) for point in val]  # a list for concatenating dimensions in each point

                new_centroid = [np.mean(dimension) for dimension in lst]

                new_centroids.append(new_centroid)

            concat_centroids = []

            for c in range(len(centroids)):

                concat_centroid = list(zip(centroids[c], new_centroids[c]))

                concat_centroids.append(concat_centroid)

            differences = []  # find the rate of change

            for item in concat_centroids:
                difference = [abs(item[n][0] - item[n][1]) for n in range(len(item))]

                differences.extend(difference)

            count_differences = np.sum([1 if diff < threshold else 0 for diff in differences])

            num_points = self.num_clusters * len(self.training_data[0])

            if (count_differences / num_points) <= 0.2:

                break


            centroids = new_centroids  # Update centroids with the new ones


        return clusters




























