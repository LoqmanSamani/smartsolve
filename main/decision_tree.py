import numpy as np


class DecisionTree:

    """
    the training_data & validation_data must both have this structure:
    list = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
    """

    def __init__(self, data, min_points=2, max_depth=2, curr_depth=0, algorithm='gini'):

        self.data = data
        self.min_points = min_points  # minimum number of points (samples) in data to split
        self.max_depth = max_depth
        self.curr_depth = curr_depth
        self.algorithm = algorithm

        self.root = None

    def train(self):
        self.root = self.build_tree(data=self.data, curr_depth=self.curr_depth)

    def build_tree(self, data, curr_depth):

        features = np.array([point[-1] for point in data])
        labels = np.array([point[0] for point in data])

        num_points = len(list(features))
        num_features = features.shape[1]

        if num_points >= self.min_points and curr_depth <= self.max_depth:

            best_split = self.best_split(
                features=features,
                num_points=num_points,
                num_features=num_features
            )

            if best_split['info_gain'] > 0:

                left_subtree = self.build_tree(
                    data=best_split['dataset_left'],
                    curr_depth=self.curr_depth + 1
                )

                right_subtree = self.build_tree(
                    data=best_split['dataset_right'],
                    curr_depth=self.curr_depth + 1
                )

                return DecisionTreeNode(
                    feature_index=best_split['feature_index'],
                    threshold=best_split['threshold'],
                    left=left_subtree,
                    right=right_subtree,
                    info_gain=best_split['info_gain']
                )

        left_value = self.left_value(labels=labels)

        return DecisionTreeNode(value=left_value)

    def best_split(self, features, num_points, num_features):

        best_split = {}
        max_info_gain = -float('inf')

        for i in range(num_features):
            feature_vals = features[:, i]
            poss_thresholds = np.unique(feature_vals)

            for t in poss_thresholds:

                dataset_left, dataset_right = self.split(
                    features=features,
                    index=i,
                    threshold=t
                )

                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    labels, l_labels, r_labels = (
                        np.array([point[-1] for point in features]),
                        np.array([point[-1] for point in dataset_left]),
                        np.array([point[-1] for point in dataset_right])
                    )

                    curr_info_gain = self.info_gain(
                        parent=labels,
                        left_child=l_labels,
                        right_child=r_labels,
                        algorithm=self.algorithm
                    )

                    if float(curr_info_gain) > max_info_gain:
                        best_split['feature_index'] = i
                        best_split['threshold'] = t
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def split(self, features, index, threshold):

        dataset_left = np.array([point for point in features if point[index] <= threshold])
        dataset_right = np.array([point for point in features if point[index] > threshold])
        return dataset_left, dataset_right

    def info_gain(self, parent, left_child, right_child, algorithm='entropy'):

        l_weight = len(left_child) / len(parent)
        r_weight = len(right_child) / len(parent)

        if algorithm == 'gini':
            gain = self.gini_impurity(
                parent=parent,
                left_child=left_child,
                right_child=right_child,
                left_weight=l_weight,
                right_weight=r_weight
            )
        else:
            gain = self.cal_entropy(
                parent=parent,
                left_child=left_child,
                right_child=right_child,
                left_weight=l_weight,
                right_weight=r_weight
            )
        return gain

    def gini_impurity(self, parent, left_child, right_child, left_weight, right_weight):

        p_gini = self.gini_index(labels=parent)
        l_gini = self.gini_index(labels=left_child) * left_weight
        r_gini = self.gini_index(labels=right_child) * right_weight

        gain = p_gini - (l_gini + r_gini)

        return gain

    def gini_index(self, labels):

        class_labels = np.unique(labels)
        gini = 0
        for label in class_labels:
            p_label = len(labels[labels == label]) / len(labels)
            gini += np.power(p_label, 2)

        return 1 - gini

    def cal_entropy(self, parent, left_child, right_child, left_weight, right_weight):

        p_entropy = self.entropy(labels=parent)
        l_entropy = self.entropy(labels=left_child) * left_weight
        r_entropy = self.entropy(labels=right_child) * right_weight

        entropy = p_entropy - (l_entropy + r_entropy)

        return entropy

    def entropy(self, labels):

        class_labels = np.unique(labels)
        entropy = 0

        for label in class_labels:

            p_label = len(labels[labels == label]) / len(labels)
            entropy += - p_label * np.log2(p_label)

        return entropy

    def left_value(self, labels):
        labels = list(labels)
        return max(labels, key=labels.count)

    def tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print('X_' + str(tree.feature_index), '<=', tree.threshold, '?', tree.info_gain)
            print('%sleft:' % (indent), end='')
            self.tree(tree.left, indent + indent)
            print('%sright' % (indent), end='')
            self.tree(tree.right, indent + indent)

    def predict(self, data):
        prediction = [self.make_prediction(point=point, tree=self.root) for point in data]
        return prediction

    def make_prediction(self, point, tree):
        if tree.value:
            return tree.value
        feature_val = point[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(point=point, tree=tree.left)
        else:
            return self.make_prediction(point=point, tree=tree.right)


class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

