from simulate import makeRandomPayoffMatrix, simulate
from player import BestResponsePlayer


# Run tests/experiments here

# Use a 3x3 payoff matrix for now
# A_ij is player 1's payoff when they choose i and player 2 chooses j
# B_ij is player 2's payoff when player 1 chooses i and player 2 chooses j
A = makeRandomPayoffMatrix(3)
B = makeRandomPayoffMatrix(3)

player1 = BestResponsePlayer(3, False)
player2 = BestResponsePlayer(3, False)
p1_totalPayoff, p2_totalPayoff, actions = simulate(player1, player2, A, B, 10)
print("Player 1 total payoff:", p1_totalPayoff)
print("Player 2 total payoff:", p2_totalPayoff)
print("Actions:", actions)