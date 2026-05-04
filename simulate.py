import numpy as np
from player import Player, BestResponsePlayer
# np.random.seed(12345) # Use a seed for debugging

# We look at payoff matrices where both players have the same number of actions
def makeRandomPayoffMatrix(numActions):
    # Make a numActions x numActions matrix of random numbers in [0, 1]
    return np.random.random((numActions, numActions))

# Note: Can use matrix multiplication if we want to calculate expected values
# For both payoff matrices, payoffs[i, j] is the payoff when player 1 plays i and player 2 plays j
def simulate(player1: Player, player2: Player, payoffs1: np.ndarray, payoffs2: np.ndarray, num_rounds: int):
    p1_totalPayoff = 0
    p2_totalPayoff = 0
    actions = []

    p1_num_actions, p2_num_actions = payoffs1.shape
    assert payoffs2.shape == (p2_num_actions, p1_num_actions)

    for t in range(num_rounds):
        probs1 = player1.chooseAction()
        choice1 = np.random.choice(p1_num_actions, p=probs1)
        probs2 = player2.chooseAction()
        choice2 = np.random.choice(p2_num_actions, p=probs2)
        player1.update(choice1, payoffs1[:,choice2])
        player2.update(choice2, payoffs2[choice1])
        p1_totalPayoff += payoffs1[choice1, choice2]
        p2_totalPayoff += payoffs2[choice1, choice2]
        actions.append((choice1, choice2))
    return p1_totalPayoff, p2_totalPayoff, actions


def run_once(p1, p2, payoffs1, payoffs2):
    p1_num_actions, p2_num_actions = payoffs1.shape
    assert payoffs2.shape == (p2_num_actions, p1_num_actions)

    # get strategies from each player
    s1 = p1.chooseAction()
    s2 = p2.chooseAction()

    # make action choices
    a1 = np.random.choice(p1_num_actions, p=s1)
    a2 = np.random.choice(p2_num_actions, p=s2)

    # update player strategies
    # p1.update(a1, payoffs1 @ s2)
    # p2.update(a2, payoffs2 @ s1)
    p1.update(a1, payoffs1[:,a2])
    p2.update(a2, payoffs2[a1])

def run(p1: Player, p2: Player, n_iterations, n_games=1, num_actions=2, **kwargs):
    # can specify game to play by providing payoff matrices for each player
    # note: this will ignore num_actions
    payoffs1: np.ndarray = kwargs.get('payoffs1', None)
    payoffs2: np.ndarray = kwargs.get('payoffs2', None)
        
    # if not specified, use random games
    random_games = payoffs1 is None or payoffs2 is None

    # use num_actions from specified game
    if not random_games:
        num1, num2 = payoffs1.shape
        assert num1 == num2, f'Both players should have the same number of actions (p1: {num1}, p2: {num2})'

        num_actions = num1
    
    # arrays to track all converged distributions (useful for single games)
    outcomes_all = []
    s1_all = []
    s2_all = []

    # run trials
    for _ in range(n_games):
        if random_games:
            payoffs1 = makeRandomPayoffMatrix(num_actions)
            payoffs2 = makeRandomPayoffMatrix(num_actions)

        # array to track distribution of outcomes at each time step
        outcomes = []

        # arrays to track strategies of each player at each time step
        s1 = []
        s2 = []

        # TODO: can also track payoffs (will be useful for tournaments)
       
        for t in range(n_iterations):
            run_once(p1, p2, payoffs1, payoffs2)

            outcomes.append(np.outer(p1.chooseAction(), p2.chooseAction()))
            s1.append(p1.chooseAction())
            s2.append(p2.chooseAction())
        
        # keep track of all time averaged distributions (only really useful if specified game)
        outcomes_all.append(np.sum(outcomes, axis=0) / n_iterations)
        s1_all.append(np.sum(s1, axis=0) / n_iterations)
        s2_all.append(np.sum(s1, axis=0) / n_iterations)
        
        # reinitialise players for new game
        p1.reset()
        p2.reset()
    
    return outcomes_all, s1_all, s2_all
