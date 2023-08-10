from matplotlib import pyplot
from time import time
import random
from argparse import ArgumentParser
from constants import PLAYER1, PLAYER2, WIN, LOSS
from q_learning_agent import QLearningAgent
from game import Game
from human_agent import HumanAgent
from random_agent import RandomAgent
import battle


# Player 1 = Q-Learning Agent
# Player 2 = Random Agent
def qlearn_vs_random(player1: QLearningAgent, games=10) -> tuple[int, int, int]:
    win, loss, draw = 0, 0, 0
    for _ in range(games):
        result = battle.faceoff(player1, RandomAgent())
        if result == WIN:
            win += 1
        elif result == LOSS:
            loss += 1
        else:
            draw += 1
    return (win / games, loss / games, draw / games)


# Command line arguments
parser = ArgumentParser(
    prog="main.py",
    description="Train a Q-Learning Agent to play Tic-Tac-Toe.",
)
parser.add_argument(
    "-g",
    "--graph",
    help="generate Q-Learning Agent vs Random Agent graph",
    action="store_true",
)
parser.add_argument(
    "-p",
    "--play",
    help="play 10 rounds against the Q-Learning Agent",
    action="store_true",
)
args = parser.parse_args()

# Setup episode numbers
EPISODES = 200_000
OUTPUT_MARKER = EPISODES / 20
METRIC_GATHER = EPISODES / 100

# Seed the randomness
random.seed(time())

# Store stats for graph generation
stats: dict = {"win": [], "loss": [], "draw": []}

# Training
player1 = QLearningAgent(PLAYER1)
player2 = QLearningAgent(PLAYER2)

for i in range(EPISODES):
    if i % OUTPUT_MARKER == 0:
        print(".", end="", flush=True)
    game = Game()
    while not game.is_over():
        action = player1.pick_and_train_action(game.board)
        game.update(action)
        if game.is_over():
            break
        action = player2.pick_and_train_action(game.board)
        game.update(action)
    player1.end_episode(game.winner)
    player2.end_episode(game.winner)

    # Save stats against Random player
    if i % METRIC_GATHER == 0:
        w, l, d = qlearn_vs_random(player1)
        stats["win"].append(w)
        stats["loss"].append(l)
        stats["draw"].append(d)

# Make stats graph for Q-Learning Agent vs Random Agent
if args.graph:
    print("Saving graph.")
    fig = pyplot.figure()
    pyplot.plot(stats["win"], label="Win")
    pyplot.plot(stats["loss"], label="Loss")
    pyplot.plot(stats["draw"], label="Draw")
    pyplot.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Score")
    pyplot.grid()
    pyplot.tight_layout()
    pyplot.legend()
    pyplot.savefig("training-" + str(int(time())) + ".png")

# Trained Q-Learning Agent vs Human
if args.play:
    print()
    print("Player 1: Human")
    print("Player 2: Q-Learning Agent")
    for i in range(5):
        print()
        print("Round", i + 1)
        result = battle.faceoff(HumanAgent(), player2)
        if result == WIN:
            print("Human wins!")
        elif result == LOSS:
            print("Q-Learning Agent wins!")
        else:
            print("Draw!")
        print()
    print("Player 1: Q-Learning Agent")
    print("Player 2: Human")
    for i in range(5):
        print()
        print("Round", i + 1)
        result = battle.faceoff(player1, HumanAgent())
        if result == WIN:
            print("Q-Learning Agent wins!")
        elif result == LOSS:
            print("Human wins!")
        else:
            print("Draw!")
        print()
