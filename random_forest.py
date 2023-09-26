import decision_tree


class RandomForest:

    def __init__(self, num_trees, max_features=None, impurity_threshold=None):

        self.num_trees = num_trees
        self.max_features = max_features
        self.impurity_threshold = impurity_threshold

        
