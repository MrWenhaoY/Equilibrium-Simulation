from matplotlib.pyplot import fill
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

class NoRegretPlayer(Player):
    def __init__(self, n_actions, learning_rate):
        self.n_actions = n_actions

        self.total_payoff = 0 
        self.iterations = 0
        
        # array of weights and resulting strategy
        self.weights = np.ones(n_actions)
        self.strategy = self.weights / n_actions
        
        # controls exploration vs exploitation
        self.learning_rate = learning_rate

        # array to keep track of fixed action payoffs
        self.fixed_payoffs = np.zeros(n_actions)

        # track total_payoff
        self.total_payoff = 0
    
    def chooseAction(self):
        return self.strategy

    def get_regret(self):
        regret = self.fixed_payoffs.max() - self.total_payoff
        return regret / self.iterations

    def update(self, action, payoffs):
        self.weights *= (1  - self.learning_rate) ** (-1 * payoffs)

        self.strategy = self.weights / self.weights.sum()
 
        # update fixed payoffs
        self.fixed_payoffs += payoffs
        self.total_payoff += payoffs[action]

        self.iterations += 1

class NoSwapPlayer(Player):
    def __init__(self, n_actions, learning_rate):
        self.n_actions = n_actions

        self.strategy =  np.full(shape=n_actions, fill_value=1/n_actions)

        self.algs = np.full(shape=n_actions, fill_value=None)

        for i in range(n_actions):
            self.algs[i] = NoRegretPlayer(n_actions, learning_rate)


    def chooseAction(self):
        return self.strategy

    def update(self, action, payoffs):
        recs = np.zeros(shape=(self.n_actions, self.n_actions))

        for i in range(self.n_actions):
            p = self.strategy[i] * payoffs
            self.algs[i].update(action, p)

            recs[i] = self.algs[i].strategy

        w, vl = eig(recs, left=True, right=False)
        idx = np.argmin(np.abs(w-1))
        pi = abs(vl[:,idx].real)
        pi = pi / pi.sum()

        self.strategy = pi
