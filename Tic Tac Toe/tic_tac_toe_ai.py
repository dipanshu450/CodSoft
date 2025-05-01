# tic_tac_toe_ai.py

import math

HUMAN = 'O'
AI = 'X'
EMPTY = ' '


def print_board(board):
    for row in board:
        print('|'.join(row))
        print('-' * 5)


def is_winner(board, player):
    # Rows, Columns, Diagonals
    return any(all(cell == player for cell in row) for row in board) or any(
        all(board[r][c] == player for r in range(3)) for c in range(3)) or all(
            board[i][i] == player  
                                      for i in range(3))


def is_board_full(board):
    return all(cell != EMPTY for row in board for cell in row)


def get_available_moves(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == EMPTY]


def minimax(board, depth, is_maximizing):
    if is_winner(board, AI):
        return 1
    elif is_winner(board, HUMAN):
        return -1
    elif is_board_full(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for (r, c) in get_available_moves(board):
            board[r][c] = AI
            score = minimax(board, depth + 1, False)
            board[r][c] = EMPTY
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for (r, c) in get_available_moves(board):
            board[r][c] = HUMAN
            score = minimax(board, depth + 1, True)
            board[r][c] = EMPTY
            best_score = min(score, best_score)
        return best_score


def best_move(board):
    best_score = -math.inf
    move = None
    for (r, c) in get_available_moves(board):
        board[r][c] = AI
        score = minimax(board, 0, False)
        board[r][c] = EMPTY
        if score > best_score:
            best_score = score
            move = (r, c)
    return move


def play_game():
    board = [[EMPTY] * 3 for _ in range(3)]
    print("Welcome to Tic-Tac-Toe! You are 'O'. AI is 'X'.")
    print_board(board)

    while True:
        # Human turn
        try:
            row = int(input("Enter row (0-2): "))
            col = int(input("Enter col (0-2): "))
        except ValueError:
            print("Invalid input. Enter numbers from 0 to 2.")
            continue

        if board[row][col] != EMPTY:
            print("That spot is already taken. Try again.")
            continue

        board[row][col] = HUMAN
        print_board(board)

        if is_winner(board, HUMAN):
            print("You win! 🎉")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

        # AI turn
        r, c = best_move(board)
        board[r][c] = AI
        print("AI has made its move:")
        print_board(board)

        if is_winner(board, AI):
            print("AI wins! 💻")
            break
        if is_board_full(board):
            print("It's a draw!")
            break


if __name__ == "__main__":
    play_game()
