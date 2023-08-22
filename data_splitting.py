class DataSplitting:

    """
    The output in each case is a list like this:
    lst = [(y1,[x11,x12,...,x1n]),(y2,[x21,x22,x2n]),...,(ym,[xm1,xm2,...,xmn])]
    """

    def __init__(self, features:dict, target:dict, method = 'RandomSplit'):

        self.features = features
        self.target = target
        self.method = method



    def __call__(self):

        if self.method == 'RandomSplit':
            return self.random_split()

        elif self.method == 'StratifiedSplit':
            return self.stratified_split()

        elif self.method == 'TimeSplit':
            return self.time_series_split()

        elif self.method == 'KfoldSplit':
            return self.cross_validation_split()

        else:
            return ("Please provide a valid splitting method. Available options are: RandomSplit,"
                    " StratifiedSplit, TimeSplit and KfoldSplit.")






    def random_split(self, test_proportion=0.2, validation_proportion=0.1)->list:

        import random as r

        target = [x for sublist in self.target.values() for x in sublist]

        features = [feature for feature in self.features.values()]

        data = list(zip(target, features))


        r.shuffle(data)
        test_data = []

        num_test = int(test_proportion * len(data))

        for i in range(num_test):

            random_number1 = r.randint(0, len(data)-1)
            item1 = data[random_number1]

            test_data.append(item1)
            data.remove(item1)


        r.shuffle(data)
        validation_data = []

        num_validation = int(validation_proportion * len(data))

        for i in range(num_validation):

            random_number2 = r.randint(0, len(data)-1)
            item2 = data[random_number2]

            validation_data.append(item2)
            data.remove(item2)


        training_data = data


        return training_data, validation_data, test_data





    def stratified_split(self, classified_data:list, test_proportion=0.2, validation_proportion=0.1)->list:

        import random as r

        classified_data = classified_data.copy()


        test_data = []

        for subset1 in classified_data:

            num_test = int(len(subset1)/len(self.target)) * test_proportion

            r.shuffle(subset1)

            for i in range(num_test):
                random_number1 = r.randint(0, len(subset1) - 1)
                item1 = subset1[random_number1]

                test_data.append(item1)
                subset1.remove(item1)


        validation_data = []

        for subset2 in classified_data:

            num_validation = int(len(subset2)/len(self.target)) * validation_proportion

            r.shuffle(subset2)

            for i in range(num_validation):

                random_number2 = r.randint(0, len(subset2) - 1)

                item2 = subset2[random_number2]

                validation_data.append(item2)
                subset2.remove(item2)

        training_data = [subset3 for subset3 in classified_data]


        return training_data, validation_data, test_data




    def time_series_split(self, date:dict, test_proportion=0.2, validation_proportion=0.1) -> list:

        target = [x for sublist in self.target.values() for x in sublist]
        features = [feature for feature in self.features.values()]
        date = [time for time in date.values()]

        data = list(zip(target, features, date))

        sorted_data = sorted(data, key=lambda x: x[-1])

        num_test = int(len(sorted_data) * test_proportion)
        num_validation = int(len(sorted_data) * validation_proportion)
        num_training = len(sorted_data) - (num_test + num_validation)

        training_data = sorted_data[: num_training]
        validation_data = sorted_data[num_training: num_training + num_validation]
        test_data = sorted_data[num_training + num_validation:]

        return training_data, validation_data, test_data



    def cross_validation_split(self, k=10, validation_proportion=0.1):

        import random as r

        target = [x for sublist in self.target.values() for x in sublist]
        features = [feature for feature in self.features.values()]

        data = list(zip(target, features))
        copied_data = data.copy()



        num_test = int(len(data) / k)
        num_validation = int(validation_proportion * len(data))

        k_sets = []
        training_data = []
        validation_data = []
        test_data = []


        for i in range(k):


            test_set = []

            while len(test_set) < num_test:

                r.shuffle(copied_data)
                random_number1 = r.randint(0, len(copied_data)-1)
                item1 = copied_data[random_number1]

                if item1 not in k_sets:

                    test_set.append(item1)
                    copied_data.remove(item1)

            test_data.append(test_set)
            k_sets.extend(test_set)


            validation_set = []

            for i in range(num_validation):

                random_number2 = r.randint(0, len(copied_data) - 1)
                item2 = copied_data[random_number2]


                validation_set.append(item2)
                copied_data.remove(item2)


            validation_data.append(validation_set)

            training_data.append(copied_data)



        return training_data, validation_data, test_data







