from decision_tree import DecisionTree
from random import sample
from time import time






class RandomForest:


    def __init__(self, training_data, feature_subsets, validation_data=None, num_trees=20, max_features=None,
                 impurity_threshold=None, data_proportion=0.20, train_time=None):
        """
        the training_data & validation_data must both have this structure:
        list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
        """
        self.training_data = training_data
        self.feature_subsets = feature_subsets
        self.validation_data = validation_data
        self.num_trees = num_trees
        self.max_features = max_features
        self.impurity_threshold = impurity_threshold
        self.data_proportion = data_proportion  # a proportion of data which will be used to train the trees
        self.train_time = train_time  # if the training process is too long, it will be stopped
        self.trees = []  # a list to store the trained trees








    def random_data_subset(self):

        num_points = int(self.data_proportion * len(self.training_data))

        data_subset = sample(self.training_data, num_points)


        return data_subset








    def predict(self, samples):

        """
        param samples:
        a nested list contains points (points = [[f1, f2, ...][f21, f22, ...], ...])
         or a single point (point = [f1, f2, ...])

        return:
        a list of predictions(numbers) or just one single number in the case of single input point.
        """

        predictions = []
        # if the input is just one point
        if len(samples) == 1:
            predictions = [tree.predict(samples) for tree in self.trees]

        # if the input is more than one point
        elif len(samples) > 1:

            for sample in samples:
                prediction = [tree.predict(sample) for tree in self.trees]

                predictions.append(prediction)

        else:
            raise ValueError("The input value should be a nested list of features"
                             " for each point of just a list of features for a single point!")

        predicted = [max(set(prediction), key=predictions.count) for prediction in predictions]

        return predicted








    def accuracy(self, test_data=None):

        if self.validation_data:

            points = [point[1] for point in self.validation_data]
            labels = [point[0] for point in self.validation_data]

            predicted = self.predict(points)

            accuracy = sum([1 if predict == label else 0 for predict, label in zip(predicted, labels)]) / len(labels)

        elif test_data:

            points = [point[1] for point in test_data]
            labels = [point[0] for point in test_data]

            predicted = self.predict(points)

            accuracy = sum([1 if predict == label else 0 for predict, label in zip(predicted, labels)]) / len(labels)

        else:
            raise ValueError("To calculate the accuracy each point must has its label!"
                             "The structure of input data must be like this: "
                             "list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]")

        return accuracy






    def train(self):

        train_time = None  # calculate the process time

        num_iterations = 0

        for _ in range(self.num_trees):

            start_time = time()

            training_data = self.random_data_subset()

            tree = DecisionTree(
                training_data=training_data,
                validation_data=self.validation_data,
                feature_subsets=self.feature_subsets,
                impurity_threshold=self.impurity_threshold
            )

            trained_tree = tree.fit()  # Capture the trained tree

            self.trees.append(trained_tree)  # Store it in the list of trees

            end_time = time()

            elapsed_time = end_time - start_time

            train_time += elapsed_time

            num_iterations += 1

            if self.train_time:

                if time > self.train_time:

                    raise TimeoutError(f"The training time is after {num_iterations} iterations out!")




































