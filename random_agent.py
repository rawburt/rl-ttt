import random
import game

class RandomAgent:
    def pick_move(self, board: list[int]) -> int:
        action = random.choice(game.available_actions(board))
        return action
