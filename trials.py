from simulate import makeRandomPayoffMatrix, simulate, run_once
from player import BestResponsePlayer, NoRegretPlayer, NoSwapPlayer
import numpy as np
import matplotlib.pyplot as plt


# Run tests/experiments here

# Use a 3x3 payoff matrix for now
# A_ij is player 1's payoff when they choose i and player 2 chooses j
# B_ij is player 2's payoff when player 1 chooses i and player 2 chooses j
A = makeRandomPayoffMatrix(3)
B = makeRandomPayoffMatrix(3)

# player1 = BestResponsePlayer(3, False)
# player2 = BestResponsePlayer(3, False)

# p1_totalPayoff, p2_totalPayoff, actions = simulate(player1, player2, A, B, 10)
# print("Player 1 total payoff:", p1_totalPayoff)
# print("Player 2 total payoff:", p2_totalPayoff)
# print("Actions:", actions)

n_games = 1
n_iterations = 100
n_actions = 3

strats1 = []
all_outcomes1 = []
for _ in range(n_games):
    # p1 = NoRegretPlayer(n_actions, 0.1)
    # p2 = NoRegretPlayer(n_actions, 0.1)
    
    p1 = NoSwapPlayer(n_actions, 0.2)
    p2 = NoSwapPlayer(n_actions, 0.2)

    # p1 = NoRegretPlayer(n_actions, 0.1)
    # p2 = BestResponsePlayer(n_actions, False)


    outcomes1 = []
    outcomes2 = []

    learning_curve = []

    game = np.array([
                    [[0,0], [-1,1], [1,-1]],
                    [[1,-1], [0,0], [-1,1]],
                    [[-1,1], [1,-1], [0,0]]
                    ])

    # game = np.array([
    #                 [[-1,-1], [-1, 0]],
    #                 [[0, -1], [-5,-5]]
    #                 ])

    A = game[:,:,0]
    B = game[:,:,1]


    for _ in range(n_iterations):
        run_once(p1, p2, A, B)
        # run_once(p3, p4, A, B)

        outcomes1.append(np.outer(p1.strategy, p2.chooseAction()))
        # outcomes2.append(np.outer(p3.strategy, p4.strategy))

    all_outcomes1.append(np.sum(outcomes1, axis=0) / n_iterations)
    # print(np.sum(outcomes1, axis = 0) / n_iterations)
    # print(np.sum(outcomes2, axis = 0))

    strats1.append(p1.strategy)
    outcomes2.append(p2.strategy)

print(np.sum(strats1, axis=0) / (n_games))
print(np.sum(all_outcomes1, axis=0) / n_games)
print(p1.strategy)

