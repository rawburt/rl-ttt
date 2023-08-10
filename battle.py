from constants import PLAYER1, WIN, LOSS, DRAW
from game import Game

# Run a game loop and return the result of the game.
def faceoff(player1, player2) -> int:
    game = Game()
    while not game.is_over():
        action = player1.pick_move(game.board)
        game.update(action)
        if game.is_over():
            break
        action = player2.pick_move(game.board)
        game.update(action)
    if game.winner:
        if game.winner == PLAYER1:
            return WIN
        else:
            return LOSS
    else:
        return DRAW
