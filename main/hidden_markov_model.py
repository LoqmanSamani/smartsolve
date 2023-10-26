import numpy as np
import pickle



class HiddenMarkovModel:
    def __init__(self, observations, num_states, num_observations=None, max_iteration=100, threshold=1e-4):
        """
        Initialize a Hidden Markov Model (HMM) instance.

        :param observations: A list or numpy array of observed values.
        :param num_states: The number of hidden states in the model.
        :param num_observations: (Optional) The number of observations (default is the length of 'observations').
        :param max_iteration: (Optional) The maximum number of training iterations (default is 100).
        :param threshold: (Optional) The convergence threshold for training (default is 1e-4).
        """

        self.observations = np.array(observations)
        self.num_states = num_states  # number of predicted states
        self.max_iteration = max_iteration
        self.threshold = threshold

        if not num_observations:
            self.num_observations = len(self.observations)  # number of observations
        else:
            self.num_observation = num_observations

        # Initialize model parameters: transition matrix, emission matrix, initial state
        self.transition_matrix = np.zeros((self.num_states, self.num_states))
        self.emission_matrix = np.zeros((self.num_states, self.num_observations))
        self.initial_state_probs = np.zeros(self.num_states)




    def model_parameters(self, transition_matrix, emission_matrix, initial_state_probs):
        """
        Set the model parameters.

        :param transition_matrix: A numpy array representing the transition probabilities between states.
        :param emission_matrix: A numpy array representing the emission probabilities of observations from states.
        :param initial_state_probs: A numpy array representing the initial state probabilities.
        """

        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_state_probs = initial_state_probs






    def train(self):
        """
        Train the Hidden Markov Model (HMM) using the Expectation-Maximization (EM) algorithm.

        :return: A message indicating the training status.
        """

        num_iteration = 1
        for _ in range(self.max_iteration):
            prev_transition_matrix = self.transition_matrix
            prev_emission_matrix = self.emission_matrix
            prev_initial_state_probs = self.initial_state_probs

            self.expectation_maximization()
            not_converged = self.check_convergence(prev_transition_matrix, prev_emission_matrix,
                                                   prev_initial_state_probs)
            if not_converged == 0:
                print(f"After {num_iteration} the model is trained and no need to more iteration!")
                break
            num_iteration += 1

        return "After full iteration the model is trained!"






    def expectation_maximization(self):
        """
        Perform one iteration of the Expectation-Maximization (EM) algorithm.

        :return: None
        """

        #  call expectation function
        forward_prob, backward_prob, total_probability = self.expectation_step()

        # update the transition matrix, emission matrix and initial state probs
        self.update_transition_matrix(forward_prob, backward_prob)
        self.update_emission_matrix(forward_prob, backward_prob)
        self.update_initial_state_probs(forward_prob)






    def update_transition_matrix(self, forward_prob, backward_prob):
        """
        Update the transition matrix based on expected counts.

        :param forward_prob: A numpy array representing forward probabilities.
        :param backward_prob: A numpy array representing backward probabilities.
        :return: None
        """

        for i in range(self.num_states):
            for j in range(self.num_states):
                # Compute the expected count of transitioning from state i to state j
                expected_count = sum(
                    forward_prob[t][i] * self.transition_matrix[i][j] *
                    self.emission_matrix[j][self.observations[t + 1]] *
                    backward_prob[t + 1][j]
                    for t in range(self.num_observations - 1)
                )

                total_expected_count = sum(
                    expected_count
                    for i in range(self.num_states)
                    for j in range(self.num_states)
                    for t in range(self.num_observations - 1)
                )

                # Update the transition matrix based on the expected count
                self.transition_matrix[i][
                    j] = expected_count / total_expected_count  # You'll need to calculate total_expected_count






    def update_emission_matrix(self, forward_prob, backward_prob):
        """
        Update the emission matrix based on expected counts.

        :param forward_prob: A numpy array representing forward probabilities.
        :param backward_prob: A numpy array representing backward probabilities.
        :return: None
        """

        for i in range(self.num_states):
            for j in range(self.num_observations):
                # Compute the expected count of being in state i and emitting observation j
                expected_count = sum(forward_prob[t][i] * backward_prob[t][i]
                                     for t in range(self.num_observations)
                                     if self.observations[t] == j
                                     )

                total_expected_count = sum(
                    sum(forward_prob[t][i] * backward_prob[t][i]
                        for t in range(self.num_observations)
                        if self.observations[t] == j)
                    for i in range(self.num_states)
                    for j in range(self.num_observations)
                )

                # Update the emission matrix based on the expected count
                self.emission_matrix[i][j] = expected_count / total_expected_count







    def update_initial_state_probs(self, forward_prob):
        """
        Update the initial state probabilities based on forward probabilities.

        :param forward_prob: A numpy array representing forward probabilities.
        :return: None
        """

        total_forward_probability = sum(forward_prob[0][i] for i in range(self.num_states))

        for i in range(self.num_states):
            # Update the initial state probability for state i based on the forward probability at the first time step
            self.initial_state_probs[i] = forward_prob[0][i] / total_forward_probability






    def check_convergence(self, prev_transition_matrix, prev_emission_matrix, prev_initial_state_probs):
        """
        Check if the model parameters have converged.

        :param prev_transition_matrix: Previous transition matrix.
        :param prev_emission_matrix: Previous emission matrix.
        :param prev_initial_state_probs: Previous initial state probabilities.
        :return: The number of non-converged elements.
        """

        previous = [prev_transition_matrix, prev_emission_matrix, prev_initial_state_probs]
        actual = [self.transition_matrix, self.emission_matrix, self.initial_state_probs]
        not_converged = 0
        for i in range(len(previous)):
            num_row, num_col = previous[i].shape
            for row in range(num_row):
                for col in range(num_col):
                    diff = abs(previous[i][row, col] - actual[i][row, col])
                    if diff > self.threshold:
                        not_converged += 1

        return not_converged






    def expectation_step(self):
        """
        Perform the expectation step of the Expectation-Maximization (EM) algorithm.

        :return: Forward probabilities, backward probabilities, and total probability.
        """

        forward_prob = np.zeros((self.num_observations, self.num_states))  # Forward probabilities
        backward_prob = np.zeros((self.num_observations, self.num_states))  # Backward probabilities

        forward_prob = self.init_forward_pass(forward_prob)  # Initialization (Forward Pass)
        forward_prob = self.rec_forward_pass(forward_prob)  # Recursion (Forward Pass)
        total_probability = self.ter_forward_pass(forward_prob)  # Termination (Forward Pass)

        backward_prob = self.init_backward_pass(forward_prob)  # Initialization (Backward Pass)
        backward_prob = self.rec_backward_pass(forward_prob)  # Recursion (Backward Pass)

        total_probability = self.normalize_total_prob(total_probability)  # normalize probabilities to average probability per observation

        return forward_prob, backward_prob, total_probability






    def init_forward_pass(self, forward_prob):
        """
        Initialize the forward probabilities for the first observation.

        :param forward_prob: A numpy array for storing forward probabilities.
        :return: Initialized forward probabilities.
        """

        for i in range(self.num_states):
            forward_prob[0][i] = self.initial_state_probs[i] * self.emission_matrix[i, self.observations[0]]

        return forward_prob




    def rec_forward_pass(self, forward_prob):
        """
        Perform the recursion step of the forward pass.

        :param forward_prob: A numpy array representing forward probabilities.
        :return: Updated forward probabilities.
        """

        for j in range(1, self.num_observations):
            for h in range(self.num_states):
                forward_prob[j, h] = np.sum(forward_prob[j - 1] * self.transition_matrix[:, h]) * self.emission_matrix[h, self.observations[h]]

        return forward_prob




    def ter_forward_pass(self, forward_prob):
        """
        Perform the termination step of the forward pass.

        :param forward_prob: A numpy array representing forward probabilities.
        :return: Total probability.
        """

        total_probability = np.sum(forward_prob[self.num_observations - 1])
        return total_probability



    def init_backward_pass(self, backward_prob):
        """
        Initialize the backward probabilities for the last observation.

        :param backward_prob: A numpy array for storing backward probabilities.
        :return: Initialized backward probabilities.
        """
        for i in range(self.num_states):
            backward_prob[self.num_observations - 1, i] = 1.0

        return backward_prob




    def rec_backward_pass(self, backward_prob):
        """
        Perform the recursion step of the backward pass.

        :param backward_prob: A numpy array representing backward probabilities.
        :return: Updated backward probabilities.
        """
        for j in range(self.num_observations - 2, -1, -1):
            for h in range(self.num_states):
                backward_prob[j, h] = np.sum(self.transition_matrix[h, :] * self.emission_matrix[:, self.observations[j + 1]] * backward_prob[j + 1, :])

        return backward_prob



    def normalize_total_prob(self, total_probability):
        """
        Normalize the total probability to average probability per observation.

        :param total_probability: Total probability.
        :return: Normalized total probability.
        """

        total_probability = total_probability / self.num_observations

        return total_probability







    def predict(self, observations):
        """
        Predict the most likely sequence of hidden states for new observations.

        :param observations: New observations.
        :return: Predicted sequence of hidden states.
        """

        # Initialize variables
        viterbi_table = self.initialize_viterbi_table(observations)
        back_pointers = self.initialize_back_pointers(observations)

        # Fill in the Viterbi table using a forward pass
        self.fill_viterbi_table(viterbi_table, back_pointers, observations)

        # Perform backtracking to find the most likely sequence of states
        predicted_states = self.back_track(back_pointers, observations)

        return predicted_states





    def initialize_viterbi_table(self, observations):
        """
        Initialize the Viterbi table for the Viterbi algorithm.

        :param observations: Observations for which Viterbi decoding is performed.
        :return: Initialized Viterbi table.
        """
        num_observations = len(observations)
        viterbi_table = np.zeros((num_observations, self.num_states))

        # Initialize the first row of the table
        for i in range(self.num_states):
            viterbi_table[0, i] = self.initial_state_probs[i] * self.emission_matrix[i, observations[0]]

        return viterbi_table





    def initialize_back_pointers(self, observations):
        """
        Initialize back pointers for the Viterbi algorithm.

        :param observations: Observations for which back pointers are initialized.
        :return: Initialized back pointers.
        """

        num_observations = len(observations)
        back_pointers = np.zeros((num_observations, self.num_states), dtype=int)

        # Initialize the back pointers for the first time step
        for i in range(self.num_states):
            back_pointers[0, i] = 0  # Initial state has no previous state

        return back_pointers





    def fill_viterbi_table(self, viterbi_table, back_pointers, observations):
        """
        Fill in the Viterbi table using a forward pass for the Viterbi algorithm.

        :param viterbi_table: Viterbi table to be filled.
        :param back_pointers: Back pointers used in the Viterbi algorithm.
        :param observations: Observations for which Viterbi decoding is performed.
        :return: None
        """

        num_observations = len(observations)

        # Iterate over time steps (observations)
        for t in range(1, num_observations):

            for current_state in range(self.num_states):

                max_score = -float('inf')
                max_prev_state = None

                # Calculate the Viterbi score for the current state at time step t
                for prev_state in range(self.num_states):

                    # Calculate the score for transitioning from prev_state to current_state
                    transition_score = viterbi_table[t - 1, prev_state] + \
                                       np.log(self.transition_matrix[prev_state, current_state])


                    # Calculate the joint score of current_state and the observation at time step t
                    score = transition_score + np.log(self.emission_matrix[current_state, observations[t]])


                    # Check if this score is greater than the current max_score
                    if score > max_score:

                        max_score = score
                        max_prev_state = prev_state


                # Update the Viterbi table and back pointers for the current state and time step
                viterbi_table[t, current_state] = max_score
                back_pointers[t, current_state] = max_prev_state





    def back_track(self, back_pointers, observations):
        """
        Perform backtracking to find the most likely sequence of hidden states.

        :param back_pointers: Back pointers used for backtracking.
        :param observations: Observations for which backtracking is performed.
        :return: Predicted sequence of hidden states.
        """

        num_observations = len(observations)

        best_path = []


        # Find the state with the highest Viterbi score at the last time step
        final_state = np.argmax(back_pointers[num_observations - 1])

        # Backtrack from the final state to the initial state
        for t in range(num_observations - 1, -1, -1):

            best_path.insert(0, final_state)  # Insert the state at the beginning of the path
            final_state = back_pointers[t, final_state]  # Move to the previous state

        return best_path



    def get_transition_matrix(self):
        """
        Get the transition matrix of the trained model.

        :return: Transition matrix.
        """
        return self.transition_matrix

    def get_emission_matrix(self):
        """
        Get the emission matrix of the trained model.

        :return: Emission matrix.
        """
        return self.emission_matrix

    def get_initial_state_probs(self):
        """
        Get the initial state probabilities of the trained model.

        :return: Initial state probabilities.
        """
        return self.initial_state_probs






    def save_model(self, filename):
        """
        Save the trained model to a file.

        :param filename: Name of the file to save the model.
        :return: None
        """

        model_params = {
            'num_states': self.num_states,
            'num_observations': self.num_observations,
            'transition_matrix': self.transition_matrix,
            'emission_matrix': self.emission_matrix,
            'initial_state_probs': self.initial_state_probs
        }

        with open(filename, 'wb') as file:
            pickle.dump(model_params, file)





    @classmethod
    def load_model(cls, filename):
        """
        Load a trained model from a file and create a new instance of the HMM class.

        :param filename: Name of the file from which to load the model.
        :return: A new instance of the HMM class with loaded model parameters.
        """

        with open(filename, 'rb') as file:
            model_params = pickle.load(file)

        num_states = model_params['num_states']
        num_observations = model_params['num_observations']
        transition_matrix = model_params['transition_matrix']
        emission_matrix = model_params['emission_matrix']
        initial_state_probs = model_params['initial_state_probs']

        # Create a new instance of the HMM class and set its parameters
        loaded_model = cls([], num_states, num_observations)
        loaded_model.transition_matrix = transition_matrix
        loaded_model.emission_matrix = emission_matrix
        loaded_model.initial_state_probs = initial_state_probs

        return loaded_model





