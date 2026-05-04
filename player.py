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

    def reset(self):
        # reset player 
        self.total_payoff = 0 # Total utility gained
        self.time = 0 # Number of time steps taken; number of actions taken


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

    def reset(self):
        super().reset()

        self.next_action = np.zeros(self.num_actions)
        if self.random:
            self.next_action += 1 / self.num_actions
        else:
            self.next_action[0] = 1


# For this class, we define the cost to be (1-payoff), where payoff is the payoff normalized between 0 and 1
class NoRegretPlayer(Player):
    def __init__(self, num_actions, learning_rate, min_val=0, max_val=1):
        super().__init__(num_actions)
        
        assert (0 < learning_rate <= 0.5), "learning_rate must be within (0, 0.5]"
        assert (min_val <= max_val), "minVal cannot be greater than maxVal"

        self.min_val = min_val
        self.max_val = max_val

        # array of weights and resulting strategy
        self.weights = np.ones(num_actions)
        self.strategy = self.weights / num_actions
        
        # controls exploration vs exploitation
        self.learning_rate = learning_rate

        # array to keep track of fixed action payoffs
        self.fixed_payoffs = np.zeros(num_actions)

        # track total_payoff
        self.total_payoff = 0
    
    def chooseAction(self):
        return self.strategy

    def get_regret(self):
        regret = self.fixed_payoffs.max() - self.total_payoff
        return regret / self.time

    def update(self, action, payoffs):
        p = 1 - (payoffs - self.min_val) / (self.max_val - self.min_val)
        self.weights *= np.pow((1 - self.learning_rate), p)

        self.strategy = self.weights / self.weights.sum()
 
        # update fixed payoffs
        self.fixed_payoffs += payoffs
        self.total_payoff += payoffs[action]

        self.time += 1

    def reset(self):
        super().reset()

        self.weights = np.ones(self.num_actions)
        self.strategy = self.weights / self.num_actions
        
        self.fixed_payoffs = np.zeros(self.num_actions)


class NoSwapPlayer(Player):
    def __init__(self, num_actions, learning_rate):
        super().__init__(num_actions)

        self.learning_rate = learning_rate

        self.strategy =  np.full(shape=num_actions, fill_value=1/num_actions)

        self.algs = np.full(shape=num_actions, fill_value=None)

        for i in range(num_actions):
            self.algs[i] = NoRegretPlayer(num_actions, learning_rate)


    def chooseAction(self):
        return self.strategy

    def update(self, action, payoffs):
        recs = np.zeros(shape=(self.num_actions, self.num_actions))

        for i in range(self.num_actions):
            p = self.strategy[i] * payoffs
            self.algs[i].update(action, p)

            recs[i] = self.algs[i].strategy

        w, vl = eig(recs, left=True, right=False)
        idx = np.argmin(np.abs(w-1))
        pi = abs(vl[:,idx].real)
        pi = pi / pi.sum()

        self.strategy = pi

    def reset(self):
        self.strategy =  np.full(shape=self.num_actions, fill_value=1/self.num_actions)

        self.algs = np.full(shape=self.num_actions, fill_value=None)

        for i in range(self.num_actions):
            self.algs[i] = NoRegretPlayer(self.num_actions, self.learning_rate)
