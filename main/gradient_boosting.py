from decision_tree import DecisionTree, DecisionTreeNode
from logistic_regression import LogisticRegression
from linear_regression import LinearRegression
from k_nearest_neighbors import KNearestNeighbors
from collections import Counter
import joblib

import numpy as np
from random import choice as ch






class GradientBoosting:


    def __init__(self, training_data, validation_data=None, n_estimators=100, learning_rate=0.1,
                 base_model='DecisionTree', threshold=1e-2, feature_subsets=None, impurity_threshold=None,
                 class_label=None, feature_idx=None, lor_threshold=0.5, lor_intercept=0, lor_algorithm='GradientDescent',
                 lr_intercept=0, lr_algorithm='GradientDescent', lr_num_parts=10, k_neighbors=5, points=None,
                 distance='Euclidean', knn_algorithm='Classification'
                 ):


        """
               Initializes the GradientBoosting model with specified parameters.

               Args:
                   training_data (list): A list of training samples, each containing a target value and a feature vector.
                   validation_data (list, optional): A list of validation samples for monitoring model performance.

                   the training_data & validation_data must both have this structure:
                   list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]

                   n_estimators (int, optional): The number of boosting iterations (weak learners).
                   learning_rate (float, optional): The learning rate, controlling the contribution of each weak learner.
                   base_model (str, optional): The base machine learning model to use (e.g., 'DecisionTree').
                   threshold (float, optional): The training stopping criterion based on loss reduction.
                   feature_subsets (dict, optional): A dictionary specifying feature subsets for diversity in DecisionTree.
                   impurity_threshold (float, optional): Threshold for splitting nodes in the DecisionTree.
                   class_label (int, optional): The target class label for LogisticRegression.
                   feature_idx (int, optional): The feature index for LogisticRegression.
                   lor_threshold (float, optional): The threshold for LogisticRegression classification.
                   lor_intercept (float, optional): The intercept term for LogisticRegression.
                   lor_algorithm (str, optional): The optimization algorithm for LogisticRegression.
                   lr_intercept (float, optional): The intercept term for LinearRegression.
                   lr_algorithm (str, optional): The optimization algorithm for LinearRegression.
                   lr_num_parts (int, optional): The number of partitions for LinearRegression.
                   k_neighbors (int, optional): The number of nearest neighbors for KNearestNeighbors.
                   points (list, optional): The data points for KNearestNeighbors.
                   distance (str, optional): The distance metric for KNearestNeighbors.
                   knn_algorithm (str, optional): The algorithm type for KNearestNeighbors ('Classification' or 'Regression').
        """



        self.training_data = training_data
        self.validation_data = validation_data
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_model = base_model
        self.threshold = threshold
        self.ensemble = []  # a list to store weak learners
        self.staged_predicted = []
        self.targets = np.array([item[0] for item in self.training_data])


        # special inputs for decision tree algorithm
        self.impurity_threshold = impurity_threshold
        self.class_label = class_label
        self.feature_inx = feature_idx
        self.feature_subsets = feature_subsets


        # special inputs for logistic regression algorithm
        self.lor_threshold = lor_threshold
        self.lor_intercept = lor_intercept
        self.lor_algorithm = lor_algorithm


        # special inputs for linear regression algorithm
        self.lr_intercept = lr_intercept
        self.lr_algorithm = lr_algorithm,
        self.lr_num_parts = lr_num_parts


        # special inputs for linear regression algorithm
        self.k_neighbors = k_neighbors
        self.points = points
        self.distance = distance
        self.knn_algorithm = knn_algorithm


        # initialize ensemble list
        if self.base_model == 'DecisionTree' or self.base_model == 'LinearRegression':
            self.ensemble = [np.mean(np.array(self.targets)) for _ in range(len(self.targets))]

        elif self.base_model == 'LogisticRegression' or self.base_model == 'KNearestNeighbors':
            target_set = list(set(self.targets))
            self.ensemble = [ch(target_set) for _ in range(len(self.targets))]







    def staged_predict(self, predicted):

        """
        Generates staged predictions for the current ensemble at each boosting iteration.
        Staged predictions can be used to monitor the model's progress during training.

        Args:
            predicted (list): The predictions made by the current ensemble for the training data.

        Returns:
            list: Staged predictions at each boosting iteration.
        """

        staged_prediction = []

        if len(self.staged_predicted) > 0:

            for pred, pre_pred, target in zip(predicted, self.staged_predicted[-1], self.targets):

                if self.base_model == 'DecisionTree' or self.base_model == 'LinearRegression' or self.knn_algorithm == 'Regression':

                    differences = [abs(pred - target), abs(pre_pred - target)]

                    if differences[0] > differences[1] or differences[0] == differences[1]:

                        staged_prediction.append(pred)

                    else:
                        staged_prediction.append(pre_pred)

                elif self.base_model == 'LogisticRegression' or self.knn_algorithm == 'Classification':

                    if pred == target:

                        staged_prediction.append(pred)
                    else:
                        staged_prediction.append(pre_pred)

        else:
            staged_prediction = predicted


        return staged_prediction








    def mean_squared_error(self, targets, predicted):
        """
        Calculates MSE (Mean Squared Error):
        MSE = (1/n) * sum((y_actual - y_predicted)^2)
        where n is the number of data points, y_actual is the actual target value,
        and y_predicted is the predicted value from the linear regression model.
        """

        targets = np.array(targets)

        predicted = np.array(predicted)

        mse = np.mean((targets - predicted) ** 2)


        return mse








    def cross_entropy_loss(self, targets, predicted):
        """
        Cross-entropy loss, or log loss, measures the performance
        of a classification model whose output is a probability
        value between 0 and 1.

        log loss = -y * log(p) - (1 - y) * log(1 - p)
        where, p is the predicted probability of the positive class,
        y is the actual label.

        """

        targets = np.array(targets)

        predicted = np.array(predicted)

        epsilon = 1e-10  # Add epsilon to avoid log(0) or log(1)

        predicted = np.clip(predicted, epsilon, 1 - epsilon)

        log_loss = - (targets * np.log(predicted) + (1 - targets) * np.log(1 - predicted))

        mean_log_loss = np.mean(log_loss)


        return mean_log_loss








    def predict(self, data):
        """
        Makes predictions for a given dataset using the trained ensemble of weak learners.

        Args:
            data (list): The dataset to make predictions on.

        Returns:
            list: Predicted values for the input dataset.
        """

        final_predictions = []

        for point in data:

            predictions = []

            # Iterate through each weak learner's contribution
            for learner_contributions in self.ensemble:

                # Calculate the predictions made by the weak learner for the input data
                learner_prediction = self.predict_by_learner(point, learner_contributions)

                predictions.append(learner_prediction)

            # Combine the predictions from all weak learners to make the final prediction
            final_prediction = self.combine_predictions(predictions)

            final_predictions.append(final_prediction)


        return final_predictions








    def predict_by_learner(self, point, learner_contributions):
        """
        Calculates a prediction for a single data point using a specific weak learner.

        Args:
            point (list): A single data point (feature vector).
            learner_contributions (array-like): Contributions of a weak learner to predictions.

        Returns:
            float: The prediction made by the specified weak learner for the input data point.
        """

        learner_prediction = None

        if self.base_model == 'LinearRegression':

            learner_prediction = np.dot(learner_contributions, point)


        elif self.base_model == 'DecisionTree':

            model = DecisionTreeNode(class_label=None, feature_idx=None)

            learner_prediction = model.predict(sample=point[1])


        elif self.base_model == 'LogisticRegression':

            model = LogisticRegression(point)

            learner_prediction = model.get_predictions()


        elif self.base_model == 'KNearestNeighbors':


            if self.knn_algorithm == 'Classification':

                model = KNearestNeighbors(point)

                _, learner_prediction = model.knn_classification()


            elif self.knn_algorithm == 'Regression':

                model = KNearestNeighbors(point)

                _, learner_prediction = model.knn_regression()



        return learner_prediction








    def combine_predictions(self, predictions):
        """
        Combines predictions from multiple weak learners to make the final ensemble prediction.

        Args:
            predictions (list): Predictions made by individual weak learners.

        Returns:
            float: The final ensemble prediction.
        """

        final_prediction = None

        if self.base_model == 'DecisionTree' or self.base_model == 'LinearRegression' or self.knn_algorithm == 'Regression':

            final_prediction = np.mean(np.array(predictions))

        elif self.base_model == 'LogisticRegression' or self.knn_algorithm == 'Classification':

            count = Counter(predictions)

            final_prediction = max(count, key=count.get)


        return final_prediction








    def train(self):
        """
        Trains the GradientBoosting model by iteratively adding weak learners to the ensemble.
        The training process continues until the stopping criterion is met or the specified number of iterations is reached.
        """

        error = np.inf  # initialize error with a big number and update it after each iteration with loss function result

        n_iter = self.n_estimators  # initialize it with number of iterations and update it after each iteration

        while error > self.threshold or n_iter > 0:

            if self.base_model == 'DecisionTree':

                # Create a new DecisionTree instance and fit it to the training data
                decision_tree = DecisionTree(training_data=self.training_data,
                                             validation_data=self.validation_data,
                                             feature_subsets=self.feature_subsets,
                                             impurity_threshold=self.impurity_threshold
                                             )


                decision_tree.fit()

                # Make predictions using the decision tree
                model = DecisionTreeNode(class_label=None, feature_idx=None)

                predicted = [model.predict(sample=point[1]) for point in self.training_data]

                staged_prediction = self.staged_predict(predicted)

                self.staged_predicted.append(staged_prediction)  # add staged_prediction in each step to self.staged_predicted to show the improvement of the model over time


                # Calculate the negative gradient of the loss function (for regression tasks)
                negative_gradient = [2 * (true_label - pred) for true_label, pred in zip(self.targets, predicted)]


                # Update the ensemble with the negative gradient
                predicted_ensemble = self.learning_rate * np.array(negative_gradient)

                self.ensemble.append(predicted_ensemble)


                # Calculate the mean squared error (MSE)
                mse = self.mean_squared_error(self.targets, predicted)

                error = mse

                n_iter -= 1


            elif self.base_model == 'LogisticRegression':

                model = LogisticRegression(training_data=self.training_data,
                                           validation_data=self.validation_data,
                                           intercept=self.lor_intercept,
                                           threshold=self.lor_threshold,
                                           algorithm=self.lor_algorithm
                                           )


                # this function from logistic regression model is the same as train()
                # and returns the predicted after train the model
                predicted = model.get_predictions()  # an array containing 0 or 1 as predicted

                """
                the LogisticRegression algorithm accepts the training data with this structure:
                list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
                
                """

                staged_prediction = self.staged_predict(predicted)

                self.staged_predicted.append(staged_prediction)


                negative_gradient = [2 * (true_label - pred) for true_label, pred in zip(self.targets, predicted)]

                predicted_ensemble = self.learning_rate * np.array(negative_gradient)

                self.ensemble.append(predicted_ensemble)


                log_loss = self.cross_entropy_loss(self.targets, predicted)

                error = log_loss

                n_iter -= 1


            elif self.base_model == 'LinearRegression':

                model = LinearRegression(train_data=self.training_data, val_data=self.validation_data,
                                         intercept=self.lr_intercept, algorithm=self.lr_algorithm,
                                         num_parts=self.lr_num_parts)


                predicted = model.get_predictions()

                staged_prediction = self.staged_predict(predicted)

                self.staged_predicted.append(staged_prediction)

                negative_gradient = [2 * (true_label - pred) for true_label, pred in zip(self.targets, predicted)]

                predicted_ensemble = self.learning_rate * np.array(negative_gradient)

                self.ensemble.append(predicted_ensemble)

                mse = self.cross_entropy_loss(self.targets, predicted)

                error = mse

                n_iter -= 1



            elif self.base_model == 'KNearestNeighbors':

                # Create a new KNearestNeighbors instance and fit it to the training data
                knn = KNearestNeighbors(training_data=self.training_data,
                                        k_neighbors=self.k_neighbors,
                                        points=self.points,
                                        distance=self.distance,
                                        algorithm=self.knn_algorithm
                                        )


                # Here, you can choose to use KNearestNeighbors for classification or regression
                if self.knn_algorithm == 'Classification':

                    _, predicted = knn.knn_classification()


                elif self.knn_algorithm == 'Regression':

                    _, predicted = knn.knn_regression()


                staged_prediction = self.staged_predict(predicted)

                self.staged_predicted.append(staged_prediction)

                negative_gradient = [2 * (true_label - pred) for true_label, pred in zip(self.targets, predicted)]

                predicted_ensemble = self.learning_rate * np.array(negative_gradient)

                self.ensemble.append(predicted_ensemble)

                # Calculate mse and log_loss
                mse = self.mean_squared_error(self.targets, predicted)

                log_loss = self.cross_entropy_loss(self.targets, predicted)


                if self.knn_algorithm == 'Classification':

                    error = log_loss

                elif self.knn_algorithm == 'Regression':

                    error = mse

                n_iter -= 1

            else:

                raise ValueError("Please provide a valid base_model. Available options are: DecisionTree, LinearRegression, LogisticRegression and KNearestNeighbors")


        return self.ensemble








    def save_model(self, filename):
        """
        Saves the trained GradientBoosting model to a file using joblib.

        Args:
            filename (str): The name of the file to save the model.
        """
        joblib.dump(self, filename)






    @classmethod
    def load_model(cls, filename):
        """
        Loads a pre-trained GradientBoosting model from a file using joblib.

        Args:
            filename (str): The name of the file containing the saved model.

        Returns:
            GradientBoosting: The loaded GradientBoosting model.
        """

        return joblib.load(filename)



