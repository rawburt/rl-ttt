from typing import Optional
from matplotlib import pyplot
from time import time
import random

EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


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
        # initialize qtable
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
        # initialize qtable
        if (state, action) not in self.qtable:
            for a in range(9):
                if (state, a) not in self.qtable:
                    self.qtable[(state, a)] = 0.05
        # epsilon greedy
        if random.random() > self.epsilon:
            scores = [self.qtable[(state, a)] for a in available_actions]
            i = scores.index(max(scores))
            action = available_actions[i]
        else:
            action = random.choice(available_actions)
        # update qtable based on last state/action pair
        if self.last_state_action:
            lstate, laction = self.last_state_action
            # the current state is the successor state we compute qval with
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
            # note: no reward is known so no reward is used
            new_qval = qval + self.learn_rate * ((self.discount * maxaq) - qval)
            self.qtable[(lstate, laction)] = new_qval
        # save chosen state/action pair
        self.last_state_action = (state, action)
        return action

    def end_episode(self, winner: Optional[int]) -> None:
        # calculate reward
        reward = 0.0
        if winner:
            if winner == self.player:
                reward = 1.0
        else:
            reward = 0.5

        # update final choice with reward
        if self.last_state_action:
            state, action = self.last_state_action
            qval = self.qtable[(state, action)]
            new_qval = qval + self.learn_rate * (reward - qval)
            self.qtable[(state, action)] = new_qval
        else:
            raise MissingLastStateActionError

        # update epsilon
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)


player1 = QLearningAgent(PLAYER1)
player2 = QLearningAgent(PLAYER2)

stats: dict = {"win": [], "draw": [], "loss": []}

# Player1 v Player2
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

    if i % 2_000 == 0:
        # Player1 v Random
        wins = 0
        draw = 0
        loss = 0
        for _ in range(10):
            game = Game()
            while not game.is_over():
                action = player1.policy_pick(game.board)
                game.update(action)
                if game.is_over():
                    break
                action = random.choice(
                    [i for i, m in enumerate(game.board) if m == EMPTY]
                )
                game.update(action)
            if game.winner:
                if game.winner == PLAYER1:
                    wins += 1
                else:
                    loss += 1
            else:
                draw += 1
        stats["win"].append(wins / 10)
        stats["draw"].append(draw / 10)
        stats["loss"].append(loss / 10)

print()
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
