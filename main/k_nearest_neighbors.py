class KNearestNeighbors:

    """
    the training_data & validation_data must both have this structure:
    list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
    points: [[x11,x12,...,x1n],y2,[x21,x22,x2n],...,[xm1,xm2,...,xmn]] without labels
    """

    def __init__(self, training_data, validation_data=None, k_neighbors=5, points=None,
                 distance='Euclidean', algorithm='Classification'):

        self.training_data = training_data
        self.validation_data = validation_data
        self.k_neighbors = k_neighbors
        self.points = points
        self.distance = distance
        self.algorithm = algorithm





    def __call__(self):

        if self.algorithm == 'Classification':
            return self.knn_classification()

        elif self.algorithm == 'Regression':
            return self.knn_regression()

        else:
            return "Please provide a valid algorithm. Available options are: Classification & Regression"





    def euclidean_distance(self):

        """
        Euclidian Distance is suitable for continuous numerical data.
        The euclidian distance between two points A and B in a multidimensional
        space is the length of the straight line connecting them. In general,
        for points given by Cartesian coordinates in n n-dimensional Euclidean
        space, the distance is:

        d(p, q) = square_root((p1 - q1)² + (p2 - q2)² + ... + (pn - qn)²)
        """

        import numpy as np

        data = self.training_data

        # d=[[[predicted1,distance1],[predicted n,distance n]],[[predicted21,distance21],[predicted2n,distance2n]],...]
        distances = []
        v_distances = []

        for point in self.points:

            point_distances = []

            for item in data:

                point_distance = sum([np.power((item[1][i] - point[i]),2) for i in range(len(item[1]))])

                point_distances.append([item[0], point_distance])

            distances.append(point_distances)



        if self.validation_data is not None:


            for point in self.validation_data:

                point_distances = []

                for item in data:

                    point_distance = sum([np.power((item[1][i] - point[1][i]), 2) for i in range(len(item[1]))])

                    point_distances.append([item[0], point_distance])

                v_distances.append(point_distances)


        return v_distances, distances






    def manhattan_distance(self):

        """
        it measures the distance between two points as the sum
        of the absolute differences between their coordinates.
        It's suitable for cases where the data lies on a grid-like structure.

        d(p, q) = |p1 - q1| + |p2 - q2| + ... + |pn - qn|
        """


        data = self.training_data

        distances = []
        v_distances = []

        for point in self.points:

            point_distances = []

            for item in data:

                point_distance = sum([abs(item[1][i] - point[i]) for i in range(len(item[1]))])

                point_distances.append([item[0], point_distance])

            distances.append(point_distances)



        if self.validation_data is not None:

            for point in self.validation_data:

                point_distances = []

                for item in data:

                    point_distance = sum([abs(item[1][i] - point[1][i]) for i in range(len(item[1]))])

                    point_distances.append([item[0], point_distance])

                v_distances.append(point_distances)



        return v_distances, distances





    def cosine_similarity(self):

        """
        Used for high-dimensional data like text or document data.
        It measures the cosine of the angle between two vectors,
        which indicates their similarity in terms of orientation.

        d(p, q) = ((p1 * q1) + (p2 * q2) + ... + (pn * qn)) / square_root((p1²+p2²+ ...+pn²)*(q1²+q2²+ ...+qn²))
        """


        import numpy as np

        data = self.training_data

        distances = []
        v_distances = []

        for point in self.points:

            point_distances = []

            for item in data:

                point_distance = (sum([(item[1][i] * point[i]) for i in range(len(item[1]))]) /
                                  np.sqrt(sum([np.power(i, 2) for i in item[1]])*sum([np.power(p, 2) for p in point])))

                point_distances.append([item[0], point_distance])

            distances.append(point_distances)



        if self.validation_data is not None:

            for point in self.validation_data:

                point_distances = []

                for item in data:

                    point_distance = (sum([(item[1][i] * point[1][i]) for i in range(len(item[1]))]) /
                                      np.sqrt(sum([np.power(i, 2) for i in item[1]]) * sum([np.power(p, 2) for p in point[1]])))

                    point_distances.append([item[0], point_distance])

                v_distances.append(point_distances)


        return v_distances, distances




    def hamming_distance(self):

        """
        Used for categorical data or binary data. It calculates the number
        of positions at which the corresponding symbols are different.
        """


        data = self.training_data

        distances = []
        v_distances = []

        for point in self.points:

            point_distances = []

            for item in data:

                point_distance = sum([1 if item[1][i] != point[i] else 0 for i in range(len(item[1]))])

                point_distances.append([item[0], point_distance])

            distances.append(point_distances)



        if self.validation_data is not None:

            for point in self.validation_data:

                point_distances = []

                for item in data:

                    point_distance = sum([1 if item[1][i] != point[1][i] else 0 for i in range(len(item[1]))])

                    point_distances.append([item[0], point_distance])

                v_distances.append(point_distances)


        return v_distances, distances



    def jaccard_similarity_index(self):

        """
        Used for sets or binary data.It calculates the size of the
        intersection of two sets divided by the size of their union.
        Often used for text analysis and recommendation systems.

        similarity(p, q) = a / (a + b + c)

        a = the number of attributes that equal 1 for both objects i and j
        b = the number of attributes that equal 0 for object i but equal 1 for object j
        c = the number of attributes that equal 1 for object i but equal 0 for object j
        """


        data = self.training_data

        similarity = []
        v_similarity = []

        for point in self.points:

            point_similarities = []

            for item in data:

                a = sum([1 if item[1][i] == 1 and point[i] == 1 else 0 for i in range(len(item[1]))])
                b = sum([1 if item[1][i] == 0 and point[i] == 1 else 0 for i in range(len(item[1]))])
                c = sum([1 if item[1][i] == 1 and point[i] == 0 else 0 for i in range(len(item[1]))])

                point_similarity = a / (a + b + c)

                point_similarities.append(point_similarity)

            similarity.append(point_similarities)



        if self.validation_data is not None:

            for point in self.validation_data:

                point_similarities = []

                for item in data:
                    a = sum([1 if item[1][i] == 1 and point[1][i] == 1 else 0 for i in range(len(item[1]))])
                    b = sum([1 if item[1][i] == 0 and point[1][i] == 1 else 0 for i in range(len(item[1]))])
                    c = sum([1 if item[1][i] == 1 and point[1][i] == 0 else 0 for i in range(len(item[1]))])

                    point_similarity = a / (a + b + c)

                    point_similarities.append(point_similarity)

                v_similarity.append(point_similarities)


        return v_similarity, similarity





    def weighted_average(self, neighbors):

        weights = [1/neighbor[1] for neighbor in neighbors]

        weighted_average = sum(weights) / len(neighbors)

        return weighted_average







    def knn_classification(self):


        import numpy as np

        targets = np.array([item[0] for item in self.training_data])
        labels = list(set(targets))



        predicted = []


        if self.distance == 'Euclidean':

            v_distances, distances = self.euclidean_distance()

        elif self.distance == 'Manhattan':

            v_distances, distances = self.manhattan_distance()

        elif self.distance == 'Cosine':

            v_distances, distances = self.cosine_similarity()

        elif self.distance == 'Hamming':

            v_distances, distances = self.hamming_distance()

        elif self.distance == 'JaccardSimilarity':

            v_distances1, distances1 = self.jaccard_similarity_index()

        else:
            return ("Please select a distance calculation method for the K-Nearest Neighbors algorithm (distance = ?)."
                    " Available methods include. The available methods are Euclidean, Manhattan, Cosine, Hamming & JaccardSimilarity")





        for point_distances in distances:

            sorted_point_distances = sorted(point_distances, key=lambda x: x[1])

            neighbors = sorted_point_distances[:self.k_neighbors]

            label_count = {}

            for i in range(len(labels)):

                count = sum([1 if labels[i] == neighbor[0] else 0 for neighbor in neighbors])

                label_count[labels[i]] = count

            max_key = max(label_count, key=label_count.get)

            predicted.append(max_key)




        for point_distances in distances1:

            sorted_point_distances = sorted(point_distances, key=lambda x: x[1])

            neighbors = sorted_point_distances[-self.k_neighbors:]

            label_count = {}

            for i in range(len(labels)):

                count = sum([1 if labels[i] == neighbor[0] else 0 for neighbor in neighbors])

                label_count[labels[i]] = count

            max_key = max(label_count, key=label_count.get)

            predicted.append(max_key)



        if self.validation_data is not None:

            validation_labels = np.array([item[0] for item in self.validation_data])

            predicted1 = []



            for point_distances in v_distances:

                sorted_point_distances = sorted(point_distances, key=lambda x: x[1])

                neighbors = sorted_point_distances[:self.k_neighbors]

                label_count = {}

                for i in range(len(labels)):
                    count = sum([1 if labels[i] == neighbor[0] else 0 for neighbor in neighbors])

                    label_count[labels[i]] = count

                max_key = max(label_count, key=label_count.get)

                predicted1.append(max_key)




            for point_distances in v_distances1:

                sorted_point_distances = sorted(point_distances, key=lambda x: x[1])

                neighbors = sorted_point_distances[-self.k_neighbors:]

                label_count = {}

                for i in range(len(labels)):
                    count = sum([1 if labels[i] == neighbor[0] else 0 for neighbor in neighbors])

                    label_count[labels[i]] = count

                max_key = max(label_count, key=label_count.get)

                predicted1.append(max_key)


            v_count = sum([1 if predicted1[i] == validation_labels[i] else 0 for i in range(len(validation_labels))])

            accuracy = round((v_count / len(validation_labels)) * 100, 2)





        return f"Based on the validation data the model accuracy is {accuracy} percent.", predicted






    def knn_regression(self):


        import numpy as np


        predicted = []


        if self.distance == 'Euclidean':

            v_distances, distances = self.euclidean_distance()

        elif self.distance == 'Manhattan':

            v_distances, distances = self.manhattan_distance()

        elif self.distance == 'Cosine':

            v_distances, distances = self.cosine_similarity()

        elif self.distance == 'Hamming':

            v_distances, distances = self.hamming_distance()

        elif self.distance == 'JaccardSimilarity':

            v_distances1, distances1 = self.jaccard_similarity_index()

        else:
            return ("Please select a distance calculation method for the K-Nearest Neighbors algorithm (distance = ?)."
                    " Available methods include. The available methods are Euclidean, Manhattan, Cosine, Hamming & JaccardSimilarity")




        for point_distances in distances:

            sorted_point_distances = sorted(point_distances, key=lambda x: x[1])

            neighbors = sorted_point_distances[:self.k_neighbors]

            weighted_average = self.weighted_average(neighbors)


            predicted.append(weighted_average)



        for point_distances in distances1:

            sorted_point_distances = sorted(point_distances, key=lambda x: x[1])

            neighbors = sorted_point_distances[-self.k_neighbors:]

            weighted_average = self.weighted_average(neighbors)

            predicted.append(weighted_average)



        if self.validation_data is not None:

            validation_labels = np.array([item[0] for item in self.validation_data])

            predicted1 = []



            for point_distances in v_distances:

                sorted_point_distances = sorted(point_distances, key=lambda x: x[1])

                neighbors = sorted_point_distances[:self.k_neighbors]

                weighted_average = self.weighted_average(neighbors)

                predicted1.append(weighted_average)



            for point_distances in v_distances1:

                sorted_point_distances = sorted(point_distances, key=lambda x: x[1])

                neighbors = sorted_point_distances[-self.k_neighbors:]

                weighted_average = self.weighted_average(neighbors)

                predicted1.append(weighted_average)




            mse = np.mean((validation_labels - predicted1) ** 2) # Mean Squared Error


        return f"Based on the validation data the model validation MSE is {mse}.", predicted

