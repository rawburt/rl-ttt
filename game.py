from typing import Optional
from constants import EMPTY, PLAYER1, PLAYER2

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


def available_actions(board: list[int]) -> list[int]:
    return [i for i, m in enumerate(board) if m == EMPTY]


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
