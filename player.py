import numpy as np
from scipy.linalg import eig

class Player:
    def __init__(self, num_actions: int):
        # payoffs is a numpy matrix 
        self.num_actions = num_actions # Number
        self.total_payoff = 0 # Total utility gained
        self.time = 0 # Number of time steps taken; number of actions taken

    def chooseAction(self)->np.ndarray:
        # Choose a distribution over the actions to play, as a np array
        choice = np.zeros(self.num_actions)
        choice[0] = 1
        return choice

    # action is the action the player chose
    # payoffs is an array containing their payoffs for each action
    def update(self, action: int, payoffs: np.ndarray):
        # opponentAction is the choice of the other player
        self.time += 1
        self.total_payoff += payoffs[action]

class BestResponsePlayer(Player):
    # If random=False: Chooses best response with lowest index
    # If radnom=True: Uniform random over best responses
    def __init__(self, num_actions: int, random: bool):
        super().__init__(num_actions)
        self.random = random
        self.next_action = np.zeros(num_actions)
        if random:
            self.next_action += 1 / num_actions
        else:
            self.next_action[0] = 1
    def chooseAction(self) -> np.ndarray:
        return self.next_action
    def update(self, action: int, payoffs: np.ndarray):
        super().update(action, payoffs)
        max_value = np.max(payoffs)

        # Choose next_action vector
        self.next_action = np.zeros(self.num_actions)
        if self.random:
            self.next_action[payoffs == max_value] = 1
            self.next_action /= np.sum(self.next_action)
        else:
            self.next_action[np.argmax(payoffs)] = 1 # This should be the first index of the maximum

# For this class, we define the cost to be (1-payoff), where payoff is the payoff normalized between 0 and 1
class MultiplicativeWeightsPlayer(Player):
    # minVal, maxVal are the maximum values that payoffs can take
    def __init__(self, num_actions: int, epsilon: float, minVal=0, maxVal=1):
        super().__init__(num_actions)
        assert (0 < epsilon <= 0.5), "epsilon must be within (0, 0.5]"
        assert (minVal <= maxVal), "minVal cannot be greater than maxVal"
        self.epsilon = epsilon
        self.weights = np.ones(num_actions) / num_actions # Normalized weights
        self.minVal = minVal
        self.maxVal = maxVal
    def chooseAction(self) -> np.ndarray:
        return self.weights
    def update(self, action: int, payoffs: np.ndarray):
        super().update(action, payoffs)
        costs = 1 - (payoffs - self.minVal) / (self.maxVal - self.minVal)
        self.weights *= np.pow((1 - self.epsilon), costs)
        
        # Normalize weights
        self.weights /= np.sum(self.weights)

class NoSwapPlayer(Player):
    def __init__(self, n_actions, learning_rate, minVal=0, maxVal=1):
        self.n_actions = n_actions

        self.weights =  np.full(shape=n_actions, fill_value=1/n_actions)

        self.algs = np.full(shape=n_actions, fill_value=None)

        for i in range(n_actions):
            self.algs[i] = MultiplicativeWeightsPlayer(n_actions, learning_rate, minVal, maxVal)


    def chooseAction(self):
        return self.weights

    def update(self, action, payoffs):
        recs = np.zeros(shape=(self.n_actions, self.n_actions))

        for i in range(self.n_actions):
            p = self.weights[i] * payoffs
            self.algs[i].update(action, p)

            recs[i] = self.algs[i].weights

        # w, vl = eig(recs, left=True, right=False)
        w, vl = eig(recs.T)
        idx = np.argmin(np.abs(w-1))
        pi = vl[:,idx].real
        pi = pi / pi.sum()

        self.weights = pi