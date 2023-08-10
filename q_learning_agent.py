from typing import Optional
from constants import EMPTY
import random


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

    # Use the Q-Table to pick a move.
    def pick_move(self, board: list[int]) -> int:
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

    # Train the Q-Table and pick a move.
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
            # NOTE: no reward is known so no reward is used
            new_qval = qval + self.learn_rate * ((self.discount * maxaq) - qval)
            self.qtable[(lstate, laction)] = new_qval
        # Save chosen state/action pair
        self.last_state_action = (state, action)
        return action

    # The end of a game when a reward is known.
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
