"""Run a game on the command line."""

import argparse

from games.snakegame import play_snake_game

parser = argparse.ArgumentParser()
parser.add_argument(
    "game",
    type=str,
    choices=["snake", "tetris", "pacman"],
    help="The name of the game to run."
)
parser.add_argument(
    "-wi",
    "--width",
    type=int,
    default=1200,
    help="The width of the game window."
)
parser.add_argument(
    "-he",
    "--height",
    type=int,
    default=800,
    help="The height of the game window."
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Run the game in debug mode."
)
args = parser.parse_args()

if args.game == "snake":
    play_snake_game(args.width, args.height, args.debug)
elif args.game == "tetris":
    pass  # TODO: Implement Tetris
elif args.game == "pacman":
    pass  # TODO: Implement Pacman
