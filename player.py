import numpy as np

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