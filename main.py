from typing import Optional
from matplotlib import pyplot
from time import time
from argparse import ArgumentParser
import random

EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


# Board state representation:
#
#   0 | 1 | 2
#   --|---|--
#   3 | 4 | 5
#   --|---|--
#   6 | 7 | 8


def winner(board: list[int]) -> Optional[int]:
    if board[0] != EMPTY:
        if (board[0] == board[1] and board[1] == board[2]) or (
            board[0] == board[3] and board[3] == board[6]
        ):
            return board[0]
    if board[4] != EMPTY:
        if (
            (board[3] == board[4] and board[4] == board[5])
            or (board[6] == board[4] and board[4] == board[2])
            or (board[0] == board[4] and board[4] == board[8])
            or (board[1] == board[4] and board[4] == board[7])
        ):
            return board[4]
    if board[8] != EMPTY:
        if (board[8] == board[7] and board[7] == board[6]) or (
            board[8] == board[5] and board[5] == board[2]
        ):
            return board[8]
    return None


def piece(p: int, pos: int) -> str:
    if p == PLAYER1:
        return "X"
    if p == PLAYER2:
        return "O"
    return str(pos)


def print_board(board: list[int]) -> None:
    print()
    print(" ", piece(board[0], 0), " | ", piece(board[1], 1), " | ", piece(board[2], 2))
    print("-----|-----|-----")
    print(" ", piece(board[3], 3), " | ", piece(board[4], 4), " | ", piece(board[5], 5))
    print("-----|-----|-----")
    print(" ", piece(board[6], 6), " | ", piece(board[7], 7), " | ", piece(board[8], 8))
    print()


class Game:
    def __init__(self) -> None:
        self.board = [EMPTY] * 9
        self.turn = PLAYER1
        self.winner: Optional[int] = None

    def is_over(self) -> bool:
        self.winner = winner(self.board)
        if self.winner:
            return True
        return self.board.count(0) == 0

    def update(self, move: int):
        self.board[move] = self.turn
        self.turn = PLAYER2 if self.turn == PLAYER1 else PLAYER1


class HumanAgent:
    def pick_move(self, board: list[int]) -> int:
        available_actions = [i for i, m in enumerate(board) if m == EMPTY]
        while True:
            print("Your move:")
            action = int(input())
            if action in available_actions:
                break
            print("Invalid pick. Try again.")
        return action


class MissingLastStateActionError(BaseException):
    pass


class QLearningAgent:
    def __init__(self, player: int) -> None:
        self.learn_rate = 0.3
        self.discount = 0.95
        self.epsilon = 0.9
        self.epsilon_decay = 0.000005
        self.epsilon_min = 0.1
        self.player = player
        self.last_state_action: Optional[tuple[tuple[int, ...], int]] = None
        self.qtable: dict = {}

    def policy_pick(self, board: list[int]) -> int:
        state = tuple(board)
        available_actions = [i for i, m in enumerate(state) if m == EMPTY]
        action = 0
        # Initialize Q-Table
        if (state, action) not in self.qtable:
            for a in range(9):
                if (state, a) not in self.qtable:
                    self.qtable[(state, a)] = 0.05
        scores = [self.qtable[(state, a)] for a in available_actions]
        i = scores.index(max(scores))
        action = available_actions[i]
        return action

    def pick_and_train_action(self, board: list[int]) -> int:
        state = tuple(board)
        available_actions = [i for i, m in enumerate(state) if m == EMPTY]
        action = 0
        # Initialize Q-Table
        if (state, action) not in self.qtable:
            for a in range(9):
                if (state, a) not in self.qtable:
                    self.qtable[(state, a)] = 0.05
        # Epsilon greedy
        if random.random() > self.epsilon:
            scores = [self.qtable[(state, a)] for a in available_actions]
            i = scores.index(max(scores))
            action = available_actions[i]
        else:
            action = random.choice(available_actions)
        # Update Q-Table based on last state/action pair
        if self.last_state_action:
            lstate, laction = self.last_state_action
            # The current state is the successor state we compute qval with
            maxaq = max(
                [
                    self.qtable[(state, 0)],
                    self.qtable[(state, 1)],
                    self.qtable[(state, 2)],
                    self.qtable[(state, 3)],
                    self.qtable[(state, 4)],
                    self.qtable[(state, 5)],
                    self.qtable[(state, 6)],
                    self.qtable[(state, 7)],
                    self.qtable[(state, 8)],
                ]
            )
            qval = self.qtable[(lstate, laction)]
            # Note: no reward is known so no reward is used
            new_qval = qval + self.learn_rate * ((self.discount * maxaq) - qval)
            self.qtable[(lstate, laction)] = new_qval
        # Save chosen state/action pair
        self.last_state_action = (state, action)
        return action

    def end_episode(self, winner: Optional[int]) -> None:
        # Calculate reward
        reward = 0.0
        if winner:
            if winner == self.player:
                reward = 1.0
        else:
            reward = 0.5

        # Update final choice with reward
        if self.last_state_action:
            state, action = self.last_state_action
            qval = self.qtable[(state, action)]
            new_qval = qval + self.learn_rate * (reward - qval)
            self.qtable[(state, action)] = new_qval
        else:
            raise MissingLastStateActionError

        # Update epsilon
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)


# Player 1 = Q-Learning Agent
# Player 2 = Random Agent
def qlearn_vs_random(player1: QLearningAgent, games=10) -> tuple[int, int, int]:
    win, loss, draw = 0, 0, 0
    for _ in range(games):
        game = Game()
        while not game.is_over():
            action = player1.policy_pick(game.board)
            game.update(action)
            if game.is_over():
                break
            action = random.choice([i for i, m in enumerate(game.board) if m == EMPTY])
            game.update(action)
        if game.winner:
            if game.winner == PLAYER1:
                win += 1
            else:
                loss += 1
        else:
            draw += 1
    return (win / games, loss / games, draw / games)


# Player 1 = Q-Learning Agent
# Player 2 = Human
def qlearn_vs_human(qplayer: QLearningAgent):
    game = Game()
    human_agent = HumanAgent()
    while not game.is_over():
        action = qplayer.policy_pick(game.board)
        game.update(action)
        if game.is_over():
            break
        print_board(game.board)
        action = human_agent.pick_move(game.board)
        game.update(action)
    if game.winner:
        print("Winner is Player ", game.winner)
    else:
        print("Draw!")


# Player 1 = Human
# Player 2 = Q-Learning Agent
def human_vs_qlearn(qplayer: QLearningAgent):
    game = Game()
    human_agent = HumanAgent()
    while not game.is_over():
        print_board(game.board)
        action = human_agent.pick_move(game.board)
        game.update(action)
        if game.is_over():
            break
        action = qplayer.policy_pick(game.board)
        game.update(action)
    if game.winner:
        print("Winner is Player ", game.winner)
    else:
        print("Draw!")


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

# Seed random library
random.seed(time())

# Training
player1 = QLearningAgent(PLAYER1)
player2 = QLearningAgent(PLAYER2)

# Store stats for graph generation
stats: dict = {"win": [], "loss": [], "draw": []}

for i in range(200_000):
    if i % 10_000 == 0:
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

    # Stats against Random player
    if i % 2_000 == 0:
        w, l, d = qlearn_vs_random(player1)
        stats["win"].append(w)
        stats["loss"].append(l)
        stats["draw"].append(d)

# Make stats graph for Q-Learning Agent vs Random Agent
if args.graph:
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
        human_vs_qlearn(player2)
        print()
    print("Player 1: Q-Learning Agent")
    print("Player 2: Human")
    for i in range(5):
        print()
        print("Round", i + 1)
        qlearn_vs_human(player1)
        print()
