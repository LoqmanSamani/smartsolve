from itertools import combinations
from collections import Counter




class AssociationRuleMining:
    """
    The structure of the input data should be like this:
    data = [
            ['item1', 'item2', 'item3'],
            ['item2', 'item4'],
            ['item1', 'item3', 'item4'],
            ...]
    Each nested list represents a transaction!
    Each item in the nested lists represents a purchase!
    """

    def __init__(self, training_data, support_threshold, confidence_threshold, algorithm='Apriori'):
        """
        Initializes an AssociationRuleMining object.

        Parameters:
            - training_data (list): The dataset used for training.
            - support_threshold (float): The minimum support threshold for frequent item sets.
            - confidence_threshold (float): The minimum confidence threshold for association rules.
            - algorithm (str): The mining algorithm to use ('Apriori' or 'FP-Growth').

        Note: 'support_threshold' and 'confidence_threshold' should be values between 0 and 1.
        """

        if not isinstance(training_data, list):
            raise ValueError("data must be a list representing your dataset.")

        if not 0 <= support_threshold <= 1:
            raise ValueError("support_threshold must be a value between 0 and 1.")

        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be a value between 0 and 1.")


        self.training_data = training_data
        self.support_threshold = support_threshold
        self.confidence_threshold = confidence_threshold
        self.algorithm = algorithm

        self.frequent_item_sets = None
        self.association_rules = None
        self.item_frequencies = None
        self.conditional_pattern_bases = None
        self.fp_tree = None







    def train(self):
        # TODO: You should implement tests and examples for your code!!!
        """
        Trains the association rule mining model based on the chosen algorithm.
        """

        self.find_frequent_item_sets(self.training_data)

        if self.algorithm == 'Apriori':

            self.association_rules = self.generate_association_rules(self.training_data, self.frequent_item_sets)








    def find_frequent_item_sets(self, data):
        """
        Finds frequent item sets based on the chosen algorithm (Apriori or FP-Growth).

        Parameters:
            - data (list): The dataset for finding frequent item sets.

        Returns:
            - None
        """

        if self.algorithm == 'Apriori':
            self.frequent_item_sets = self.apriori_algorithm(data)

        elif self.algorithm == 'FP-Growth':
            self.fp_tree = self.build_fp_tree(self.training_data)

        else:
            raise ValueError("This model supports 'Apriori Algorithm' and 'FP-Growth',"
                             " please define one of them as algorithm.")







    def apriori_algorithm(self, data):
        """
        Implements the Apriori algorithm to find frequent item sets.

        Parameters:
            - data (list): The dataset for finding frequent item sets.

        Returns:
            - frequent_item_sets (list of lists): List of frequent item sets.
        """

        frequent_item_sets = []  # List to store frequent item_sets
        frequent_item_sets_k_minus_1 = []  # Initialize for the first iteration
        k = 1

        while True:

            frequent_item_sets_k = self.generate_frequent_item_sets(data, k, frequent_item_sets_k_minus_1)

            if not frequent_item_sets_k:
                break

            frequent_item_sets.extend(frequent_item_sets_k)

            # Update frequent_item_sets_k_minus_1 for the next iteration
            frequent_item_sets_k_minus_1 = frequent_item_sets_k

            # Generate candidate item_sets of size k+1
            k += 1

        return frequent_item_sets







    def generate_frequent_item_sets(self, data, k, frequent_item_sets_k_minus_1):
        """
        Generates frequent item sets of size k based on candidate item sets of size k+1.

        Parameters:
            - data (list): The dataset for finding frequent item sets.
            - k (int): The size of the item sets to generate.
            - frequent_item_sets_k_minus_1 (list of lists): List of frequent item sets of size k-1.

        Returns:
            - frequent_item_sets (list of lists): List of frequent item sets of size k.
        """

        candidate_item_sets = {}
        frequent_item_sets = []

        for transaction in data:
            # Generate combinations of k+1 items
            subsets = list(combinations(transaction, k + 1))

            for subset in subsets:
                # Convert subset to a sorted tuple to use it as a dictionary key
                subset_key = tuple(sorted(subset))

                if subset_key in candidate_item_sets:
                    candidate_item_sets[subset_key] += 1
                else:
                    candidate_item_sets[subset_key] = 1

        for item_set, count in candidate_item_sets.items():
            support = count / len(data)

            # Check if all subsets of size k are frequent
            all_subsets_frequent = all(
                tuple(sorted(subset)) in frequent_item_sets_k_minus_1 for subset in combinations(item_set, k))

            if support >= self.support_threshold and all_subsets_frequent:
                frequent_item_sets.append(list(item_set))

        return frequent_item_sets







    def generate_association_rules(self, data, frequent_item_sets):
        """
        Generates association rules from frequent item sets (Apriori-specific).

        Parameters:
            - data (list): The dataset for generating association rules.
            - frequent_item_sets (list of lists): List of frequent item sets.

        Returns:
            - association_rules (list of tuples): List of association rules (antecedent, consequent, confidence).
        """

        association_rules = []
        for item_set in frequent_item_sets:

            sub_item_sets = [set(list(combinations(item_set, i))) for i in range(len(item_set)-1)]

            for item in sub_item_sets:

                antecedent = [item]
                consequent = [x for x in item_set if x != item]
                confidence = self.calculate_confidence(data, antecedent, consequent)

                if confidence >= self.confidence_threshold:

                    association_rules.append((antecedent, consequent, confidence))

        return association_rules








    def calculate_confidence(self, data, antecedent, consequent):
        """
        Calculates the confidence of an association rule.

        Parameters:
            - data (list): The dataset for calculating confidence.
            - antecedent (list): The antecedent of the association rule.
            - consequent (list): The consequent of the association rule.

        Returns:
            - confidence (float): The confidence score of the association rule.
        """

        antecedent_support = sum(1 for transaction in data if set(antecedent).issubset(transaction))
        combined_support = sum(1 for transaction in data if set(antecedent + consequent).issubset(transaction))

        if antecedent_support == 0:
            return 0.0  # Avoid division by zero

        return combined_support / antecedent_support








    def count_item_frequencies(self, data):
        """
        Counts the frequency of each item in the dataset.

        Parameters:
            - data (list): The dataset for counting item frequencies.

        Returns:
            - item_frequencies (Counter): A Counter object with item frequencies.
        """

        all_items = []

        for transaction in data:

            all_items.extend(transaction)

        item_frequencies = Counter(all_items)

        return item_frequencies









    def build_conditional_pattern_base(self, data, item):
        """
        Builds a conditional pattern base for a given item (FP-Growth-specific).

        Parameters:
            - data (list): The dataset for building the conditional pattern base.
            - item (str): The item for which the conditional pattern base is built.

        Returns:
            - conditional_pattern_base (list of lists): Conditional pattern base for the item.
        """

        conditional_pattern_base = []
        for i in range(len(data)):
            if item in data[i]:
                conditional_pattern_base.append(data[i])

        return conditional_pattern_base







    def encode_transactions(self, conditional_pattern_bases):
        """
        Encodes transactions using item frequencies (FP-Growth-specific).

        Parameters:
            - conditional_pattern_bases (dict): Dictionary of item-to-conditional pattern bases.

        Returns:
            - encoded_transactions (dict): Dictionary of item-to-encoded transactions.
        """

        encoded_transactions = {}

        for item, transactions in conditional_pattern_bases.items():

            encoded_transactions[item] = []

            item_frequencies = self.count_item_frequencies(transactions)

            for transaction in transactions:

                # Sort items by frequency in descending order
                sorted_items = sorted(transaction, key=lambda x: item_frequencies[x], reverse=True)

                # Encode items using their frequencies
                encoded_transaction = [item_frequencies[item] for item in sorted_items]

                encoded_transactions[item].append(encoded_transaction)

        return encoded_transactions








    def sort_items_by_frequency(self, item_frequencies):
        """
        Sorts items by their frequency in descending order (FP-Growth-specific).

        Parameters:
            - item_frequencies (Counter): Counter object with item frequencies.

        Returns:
            - sorted_items (list): List of items sorted by frequency.
        """

        sorted_items = sorted(item_frequencies.keys(), key=lambda x: item_frequencies[x], reverse=True)

        return sorted_items







    def build_fp_tree_structure(self, encoded_transactions, sorted_items):
        """
        Builds the FP-tree structure based on encoded transactions (FP-Growth-specific).

        Parameters:
            - encoded_transactions (dict): Dictionary of item-to-encoded transactions.
            - sorted_items (list): List of items sorted by frequency.

        Returns:
            - fp_tree (dict): The constructed FP-tree structure.
        """

        fp_tree = {}  # Initialize an empty FP-tree

        for item in sorted_items:

            for encoded_transaction in encoded_transactions[item]:

                subtree = fp_tree

                for node in encoded_transaction:

                    if node not in subtree:

                        subtree[node] = {}  # Create a new node if it doesn't exist

                    subtree = subtree[node]  # Move to the next node


        return fp_tree








    def build_fp_tree(self, data):
        """
        Builds the FP-tree and prunes items that don't meet the support threshold (FP-Growth-specific).

        Parameters:
            - data (list): The dataset for building the FP-tree.

        Returns:
            - fp_tree (dict): The constructed FP-tree structure.
        """

        conditional_pattern_bases = {}

        item_frequencies = self.count_item_frequencies(data)

        self.item_frequencies = item_frequencies


        for item in item_frequencies.keys():

            conditional_pattern_base = self.build_conditional_pattern_base(data, item)
            conditional_pattern_bases[item] = conditional_pattern_base


        for item, transactions in conditional_pattern_bases.items():

            frequency = len(transactions) / len(data)

            if frequency <= self.support_threshold:

                del conditional_pattern_bases[item]

        self.conditional_pattern_bases = conditional_pattern_bases

        encoded_transactions = self.encode_transactions(conditional_pattern_bases)

        sorted_items = self.sort_items_by_frequency(item_frequencies)

        fp_tree = self.build_fp_tree_structure(encoded_transactions, sorted_items)

        return fp_tree







    def get_frequent_item_sets(self):
        """
        Returns the frequent item sets (for Apriori).

        Returns:
            - frequent_item_sets (list of lists): List of frequent item sets.
        """

        if self.algorithm == 'Apriori':
            return self.frequent_item_sets
        else:
            raise ValueError('This parameter is not calculated because the used algorithm was not Apriori')







    def get_association_rules(self):
        """
        Returns the association rules (for Apriori).

        Returns:
            - association_rules (list of tuples): List of association rules (antecedent, consequent, confidence).
        """

        if self.algorithm == 'Apriori':

            return self.association_rules

        else:

            raise ValueError('This parameter is not calculated because the used algorithm was not Apriori')







    def get_item_frequencies(self):
        """
        Returns item frequencies (for FP-Growth).

        Returns:
            - item_frequencies (Counter): Counter object with item frequencies.
        """

        if self.algorithm == 'FP-Growth':

            return self.item_frequencies
        else:
            raise ValueError('This parameter is not calculated because the used algorithm was not FG-Growth')







    def get_conditional_pattern_bases(self):
        """
        Returns conditional pattern bases (for FP-Growth).

        Returns:
            - conditional_pattern_bases (dict): Dictionary of item-to-conditional pattern bases.
        """

        if self.algorithm == 'FP-Growth':

            return self.conditional_pattern_bases
        else:
            raise ValueError('This parameter is not calculated because the used algorithm was not FG-Growth')






    def get_fp_tree(self):
        """
        Returns the FP-tree structure (for FP-Growth).

        Returns:
            - fp_tree (dict): The constructed FP-tree structure.
        """

        if self.algorithm == 'FP-Growth':

            return self.fp_tree
        else:
            raise ValueError('This parameter is not calculated because the used algorithm was not FG-Growth')






