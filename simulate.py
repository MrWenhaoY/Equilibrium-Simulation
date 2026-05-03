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

