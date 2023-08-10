import random
import game

class RandomAgent:
    # Choose a random available open splot on the board.
    def pick_move(self, board: list[int]) -> int:
        action = random.choice(game.available_actions(board))
        return action
