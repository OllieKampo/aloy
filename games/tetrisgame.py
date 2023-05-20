import curses
from random import randint

# Initialize the screen
stdscr = curses.initscr()
curses.curs_set(0)  # Hide the cursor
stdscr.nodelay(1)  # Non-blocking input
stdscr.timeout(100)  # Set refresh rate to 100 milliseconds

# Set up the game board
board_width = 10
board_height = 20
board = [[' ' for _ in range(board_width)] for _ in range(board_height)]

# Tetromino shapes and their respective colors
tetrominos = [
    [['I'],
     ['I'],
     ['I'],
     ['I']],

    [['O', 'O'],
     ['O', 'O']],

    [['J', ' '],
     ['J', 'J'],
     ['J', ' ']],

    [[' ', 'L'],
     ['L', 'L'],
     [' ', 'L']],

    [['T', 'T', 'T'],
     [' ', 'T', ' ']],

    [['Z', 'Z', ' '],
     [' ', 'Z', 'Z']],

    [[' ', 'S', 'S'],
     ['S', 'S', ' ']]
]

tetromino_colors = [curses.COLOR_CYAN, curses.COLOR_YELLOW, curses.COLOR_BLUE,
                    curses.COLOR_MAGENTA, curses.COLOR_GREEN, curses.COLOR_RED,
                    curses.COLOR_WHITE]

index = randint(0, len(tetrominos) - 1)

# Set up game variables
current_piece = tetrominos[index]
current_piece_color = tetromino_colors[index]
current_piece_x = board_width // 2 - 1
current_piece_y = 0

score = 0


# Function to draw the game board
def draw_board() -> None:
    stdscr: curses.window = curses.initscr()
    stdscr.clear()
    stdscr.border(0)
    for y, row in enumerate(board):
        for x, char in enumerate(row):
            stdscr.addch(y + 1, x + 1, char)
    stdscr.addstr(board_height + 2, 2, f"Score: {score}")


def is_valid_position(piece: list[str], x: int, y: int) -> bool:
    """Check if a position is valid for the current piece."""
    for row in range(len(piece)):
        for col in range(len(piece[row])):
            if (
                piece[row][col] != ' ' and
                (board[y + row][x + col] != ' ' or
                 y + row >= board_height or
                 x + col < 0 or x + col >= board_width)
            ):
                return False
    return True


# Function to rotate a piece
def rotate_piece(piece):
    return list(zip(*reversed(piece)))

# Function to place the current piece on the board
def place_piece():
    global current_piece, current_piece_color, current_piece_x, current_piece_y, score

    for row in range(len(current_piece)):
        for col in range(len(current_piece[row])):
            if current_piece[row][col] != ' ':
                board[current_piece_y + row][current_piece_x + col] = current_piece[row][col]

    # Check for completed rows
    for row in range(board_height):
        if all(cell != ' ' for cell in board[row]):
            score += 1
            del board[row]
            board.insert(0, [' '] * board_width)

    # Reset current piece
    index = randint(0, len(tetrominos) - 1)
    current_piece = tetrominos[index]
    current_piece_color = tetromino_colors[index]
    current_piece_x = board_width // 2 - 1
    current_piece_y = 0


# Main game loop
while True:
    # Get user input
    key = stdscr.getch()

    if key == ord('q'):
        break  # Exit the game

    if key == curses.KEY_LEFT and is_valid_position(current_piece, current_piece_x - 1, current_piece_y):
        current_piece_x -= 1

    if key == curses.KEY_RIGHT and is_valid_position(current_piece, current_piece_x + 1, current_piece_y):
        current_piece_x += 1

    if key == curses.KEY_DOWN and is_valid_position(current_piece, current_piece_x, current_piece_y + 1):
        current_piece_y += 1

    if key == ord(' '):
        rotated_piece = rotate_piece(current_piece)
        if is_valid_position(rotated_piece, current_piece_x, current_piece_y):
            current_piece = rotated_piece

    # Move the current piece down
    if is_valid_position(current_piece, current_piece_x, current_piece_y + 1):
        current_piece_y += 1
    else:
        place_piece()

    # Draw the game board
    draw_board()

# Clean up and exit
curses.endwin()
