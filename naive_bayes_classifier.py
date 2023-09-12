
class NaiveBayesClassifier:
    def __init__(self, training_data, validation_data=None, laplace_smoothing=1e-5):

        """
        The input training data must be like this : [[label,[a text which is already split into words]],[...],...]
        """

        self.training_data = training_data
        self.validation_data = validation_data
        self.laplace_smoothing = laplace_smoothing

        self.label_probability = None   # {label1:probability, label2:probability, ...}
        self.label_word_probability = []   # lst = [(label1,{word1:probability, words2:probability,...}),(label2,{...})]




    def prior_probability(self, labels):
        """
        It calculates the probability of each class in training data
        """
        specific_labels = list(set(labels))

        label_probability = {i: labels.count(i)/len(labels) for i in specific_labels}

        return label_probability


    def word_probability(self, text, word):
        """
        it calculates the probability of a given word in a class
        """
        count_word = text.count(word) + self.laplace_smoothing

        word_probability = count_word/len(text) + self.laplace_smoothing

        return word_probability


    def split_data(self):
        """
        it separates each text based on its label
        """

        separated_texts = {}

        labels = list(set([item[0] for item in self.training_data]))

        for label in labels:

            separated_texts[label] = [text for text in self.training_data if text[0] == label]

        return separated_texts


    def train(self):

        labels = [item[0] for item in self.training_data]

        separated_texts = self.split_data()

        self.label_probability = self.prior_probability(labels)

        for label, texts in separated_texts.items():

            words_probabilities = {}
            for text in texts:
                words_probability = {}

                for word in text:

                    word_probability = self.word_probability(text, word)

                    if word not in words_probability:

                        words_probability[word] = word_probability

                for key in words_probability.keys():

                    if key not in words_probabilities:

                        words_probabilities[key] = words_probability
                    else:
                        words_probabilities[key] += words_probability

            self.label_word_probability.append((label, words_probabilities))

        accuracy = None

        if self.validation_data:

            v_labels = [item[0] for item in self.validation_data]
            predicted = self.validation()

            accuracy = (sum([1 for i, j in zip(v_labels, predicted) if i == j]) / len(v_labels)) ** 100

        return f"After training the model can predict(based on validation data) new text with {accuracy} percent."








    def predict(self, instances):

        predicted_labels = []

        for instance in instances:

            text_probabilities = []

            for i in range(len(self.label_word_probability)):

                word_probability = None

                for word in instance:

                    if word in self.label_word_probability[i][1].keys():

                        word_probability += self.label_word_probability[i][1][word]

                    else:

                        word_probability += self.laplace_smoothing

                text_probabilities.append(word_probability)

            all_pro = {}  # each key-value pair contains label and its probability

            pro = [text * prob for text, prob in zip(text_probabilities, self.label_probability.values())]

            for j in range(len(self.label_probability)):

                 all_pro[self.label_probability.keys()[j]] = pro[j]

            predicted_label = max(all_pro, key=lambda x: all_pro[x])

            predicted_labels.append(predicted_label)

        return predicted_labels





    def validation(self):

        predicted_labels = []

        validation_lst = [item[1] for item in self.validation_data]

        for text in validation_lst:

            for i in range(len(self.label_word_probability)):

                text_probabilities = []

                word_probability = None

                for word in text:

                    if word in self.label_word_probability[i][1].keys():

                        word_probability += self.label_word_probability[i][1][word]

                    else:

                        word_probability += self.laplace_smoothing

                text_probabilities.append(word_probability)

                all_pro = {}  # each key-value pair contains label and its probability

                pro = [text * prob for text, prob in zip(text_probabilities, self.label_probability.values())]

                for j in range(len(self.label_probability)):

                    all_pro[self.label_probability.keys()[j]] = pro[j]

                predicted_label = max(all_pro, key=lambda x: all_pro[x])

                predicted_labels.append(predicted_label)


        return predicted_labels






































