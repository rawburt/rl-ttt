import game

class HumanAgent:
    # Display the board state and get input from the terminal.
    # A valid selection of an open slot on the board is required.
    def pick_move(self, board: list[int]) -> int:
        game.print_board(board)
        available_actions = game.available_actions(board)
        while True:
            print("Your move:")
            action = int(input())
            if action in available_actions:
                break
            print("Invalid pick. Try again.")
        return action
