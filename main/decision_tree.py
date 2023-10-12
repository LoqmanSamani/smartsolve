
class DecisionTree:

    """
    the training_data & validation_data must both have this structure:
    list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
    """

    def __init__(self, training_data, validation_data=None, feature_subsets, impurity_threshold=None):

        self.training_data = training_data
        self.validation_data = validation_data
        self.feature_subsets = feature_subsets # a dictionary contains all features and diversity of them for example: dict = {feature1:[1,2,3], feature2:[1,2,3,4], ...}
        self.impurity_threshold = impurity_threshold
        self.tree = None  # Placeholder for the decision tree structure



    def gini_impurity(self, features):

        import numpy as np

        len_data = np.sum([len(subfeature) for subfeature in features[0]])

        gini_imps = []

        for feature in features:

            gini_imp = 1 - np.sum([np.power((len(sub_feature) / len_data), 2) for subfeature in feature])

            gini_imps.append(gini_imp)

        best_f_to_split_on = np.min(gini_imps)


        return best_f_to_split_on




    def child_node_impurities(self, features):

        import numpy as np

        len_data = np.sum([len(subfeature) for subfeature in features[0]])

        gini_imps = []

        for feature in features:

            gini_imp = 1 - np.sum([np.power((len(sub_feature) / len_data), 2) for subfeature in feature])

            gini_imps.append(gini_imp)


        return gini_imps




    def parent_impurity(self, feature):

        import numpy as np

        len_data = np.sum([len(subfeature) for subfeature in feature[0]])

        gini_imp = 1 - np.sum([np.power((len(subfeature) / len_data), 2) for subfeature in feature])


        return gini_imp








    def data_splitting(self, data):

        import numpy as np

        features = [] # a list of all features, each feature contains sublists of all categories in the feature.

        for i in range(len(self.feature_subsets)):  # number of features

            feature = []

            for value in self.feature_subsets.values():  # each feature

                subfeature = []

                for item in value:

                    for point in data:

                        if item == point[1][i]:

                            subfeature.append(point)

                feature.append(subfeature)

            features.append(feature)

        parent_impurity_val = self.parent_impurity(data)

        child_node_impurities = self.child_node_impurities(features)

        childs = self.data_splitting(features)

        weighted_child_node_impurities = np.mean([len(child) / len(data) for child in childs])

        impurity_reduction = parent_impurity_val - weighted_child_node_impurities



        return childs





    def get_most_common_class(self, data):

        from collections import Counter

        class_labels = [point[0] for point in data]

        class_counts = Counter(class_labels)

        most_common_class = class_counts.most_common(1)[0][0]


        return most_common_class






    def get_best_feature_to_split_on(self, features):

        import numpy as np

        len_data = np.sum([len(subfeature) for subfeature in features[0]])

        gini_imps = []

        for feature in features:

            gini_imp = 1 - np.sum([np.power((len(sub_feature) / len_data), 2) for subfeature in feature])

            gini_imps.append(gini_imp)


        best_f_to_split_on = np.min(gini_imps)


        return best_f_to_split_on






    def tree_building(self, data=None):

        if data is None:

            data = self.training_data

        if self.impurity_threshold is not None:

            parent_impurity_val = self.parent_impurity(data)

            if parent_impurity_val <= self.impurity_threshold:

                # Create a leaf node with the most common class label
                most_common_class = self.get_most_common_class(data)


                return DecisionTreeNode(class_label=most_common_class)


        splited_data = self.data_splitting(data)

        if not splited_data:  # If no further split can be done

            most_common_class = self.get_most_common_class(data)


            return DecisionTreeNode(class_label=most_common_class)

        feature_idx_to_split_on = self.get_best_feature_to_split_on(splited_data)

        feature_subsets = self.feature_subsets[feature_idx_to_split_on]

        node = DecisionTreeNode(feature_idx=feature_idx_to_split_on)

        for value in feature_subsets:

            child_data = [point for point in data if point[1][feature_idx_to_split_on] == value]

            child_tree = self.tree_building(child_data)

            node.add_child(value, child_tree)  # Add child node to the current node


        return node




    def fit(self):

        self.tree = self.tree_building()

        return self.tree  # Return the trained tree






class DecisionTreeNode:


    def __init__(self, class_label=None, feature_idx=None):
        self.class_label = class_label  # Class label for leaf nodes
        self.feature_idx = feature_idx  # Index of the feature to split on for non-leaf nodes
        self.children = {}  # Dictionary to hold child nodes

    def add_child(self, value, child_node):

        self.children[value] = child_node


    def is_leaf(self):

        return self.class_label is not None




    def predict(self, sample):

        if self.is_leaf():

            return self.class_label

        value = sample[self.feature_idx]

        if value in self.children:

            child_node = self.children[value]

            return child_node.predict(sample)

        else:

            return 'There is an issue!!!'







