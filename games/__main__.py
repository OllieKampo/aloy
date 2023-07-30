"""Run a game on the command line."""

import argparse
import importlib
import importlib.util
import sys
from typing import Any, Callable, Final, Iterable, NamedTuple

import curses
import pyfiglet

from auxiliary.argparseutils import mapping_argument_factory


class GameParam(NamedTuple):
    """A class to store information about a game parameter."""

    names: list[str]
    type_: type
    default: Any
    help: str


def _optional_int(value: str) -> int | None:
    """Convert a string to an integer if possible, otherwise return None."""
    try:
        return int(value)
    except ValueError:
        return None


__STANDARD_PARAMS: Final[list[GameParam]] = [
    GameParam(
        ["-w", "--width"],
        _optional_int,  # type: ignore
        None,
        "The width of the game window, each game has its own default width."
    ),
    GameParam(
        ["-t", "--height"],
        _optional_int,  # type: ignore
        None,
        "The height of the game window, each game has its own default height."
    ),
    GameParam(
        ["--debug"],
        bool,
        False,
        "Run the game in debug mode."
    )
]


def add_standard_parameters(parser: argparse.ArgumentParser) -> None:
    """Add standard parameters to an argument parser."""
    for parameter in __STANDARD_PARAMS:
        parser.add_argument(
            *parameter.names,
            type=parameter.type_,
            default=parameter.default,
            help=parameter.help
        )
    parser.add_argument(
        "-params",
        nargs="*",
        action=mapping_argument_factory(),
        type=str,
        metavar="KEY_1=VALUE_1 KEY_i=VALUE_i [...] KEY_n=VALUE_n",
        help="Parameters for the game."
    )


__NAME_RESERVED_CHARACTERS: Final[set[str]] = {"-", "_", " "}


class GameRegistration(NamedTuple):
    """Tuple to store information about a game registration."""

    module: str
    entry_point: str
    name: str
    description: str
    default_size: tuple[int, int]
    competitive: bool
    parameters: list[GameParam]
    package: str = "games"


__registered_games__: dict[str, GameRegistration] = {}


def register_game(
    module: str,
    entry_point: str,
    name: str,
    description: str,
    default_size: tuple[int, int],
    competitive: bool = False,
    parameters: Iterable[GameParam] | None = None,
    package: str = "games"
) -> None:
    """
    Register a game with the game launcher.

    Parameters
    ----------
    `module: str` - The name of the module containing the game.

    `entry_point: str` - The name of the function to call to run the game.

    `name: str` - The name of the game.

    `description: str` - A description of the game. This will be displayed
    when the user lists the available games.

    `default_size: tuple[int, int]` - The default size of the game window in
    pixels (width, height).

    `competitive: bool = False` - Whether the game is competitive. If the game
    is competitive, the game launcher will ask the user if they want to play
    against the computer or another player.

    `parameters: dict[str, GameParam] | None = None` - A dictionary of
    parameters for the game. The keys are the names of the parameters, and the
    values are `GameParam` objects, containing the type, default value, and
    help text for the parameter. The keys must be valid Python identifiers,
    and must not be the same as any of the standard parameters, see
    `aloy.games.standard_params()`. Note that the parameters will be passed
    to the game entry point as keyword arguments.

    `package: str = "games"` - The name of the package containing the game
    module if it is not in the root package.
    """
    # Check if the module exists.
    spec = importlib.util.find_spec(f".{module}", package=package)
    if spec is None:
        raise ValueError(f"Module '{module}' does not exist.")

    # Check name is valid.
    if name in __registered_games__:
        raise ValueError(f"Game '{name}' is already registered.")
    for char in __NAME_RESERVED_CHARACTERS:
        if char in name:
            raise ValueError(
                f"Game name '{name}' contains reserved character '{char}'."
            )

    # Check size is valid.
    if len(default_size) != 2:
        raise ValueError(
            f"Default size '{default_size}' is not a tuple of length 2."
        )
    for value in default_size:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                "Default size must be a tuple of positive integers. "
                f"Got; '{default_size}'."
            )

    # Check parameters are valid.
    if parameters is None:
        parameters = []
    for parameter in parameters:
        for name in parameter.names:
            for standard_parameter in __STANDARD_PARAMS:
                if name in standard_parameter.names:
                    raise ValueError(
                        f"Game parameter '{name}' is a standard parameter."
                    )
    parameters = list(parameters)

    __registered_games__[name] = GameRegistration(
        entry_point=entry_point,
        module=module,
        name=name,
        description=description,
        default_size=default_size,
        competitive=competitive,
        parameters=parameters,
        package=package
    )


register_game(
    "snakegame",
    "play_snake_game",
    "Snake",
    "Play the snake game with Qt.",
    default_size=(800, 600)
)


register_game(
    "tetrisgame",
    "play_tetris_game",
    "Tetris",
    "Play the tetris game with Qt.",
    default_size=(800, 800)
)


# register_game(
#     "tetrisgame",
#     "play_tetris_curses_game",
#     "TetrisCurses",
#     "Play the tetris game with Curses."
# )


# register_game(
#     "blockbreakergame",
#     "play_blockbreaker_game",
#     "Block Breaker",
#     "Play the block breaker game."
# )

# register_game(
#     "pacmangame",
#     "play_pacman_game",
#     "Pacman",
#     "Play the pacman game."
# )

# register_game(
#     "ponggame",
#     "play_pong_game",
#     "Pong",
#     "Play the pong game.",
#     competitive=True
# )

# register_game(
#     "connectfourgame",
#     "play_connect_four_game",
#     "Connect Four",
#     "Play the connect four game.",
#     competitive=True
# )

# register_game(
#     "chessgame",
#     "play_chess_game",
#     "Chess",
#     "Play the chess game.",
#     competitive=True
# )

# register_game(
#     "gogame",
#     "play_go_game",
#     "Go",
#     "Play the go game.",
#     competitive=True
# )


def run_game_launcher(stdscr: curses.window) -> str | None:
    """Run a game launcher with curses."""
    # stdscr: curses.window = curses.initscr()

    curses.noecho()
    curses.cbreak()
    curses.curs_set(False)
    stdscr.keypad(True)
    stdscr.clear()
    stdscr.border(0)
    stdscr.refresh()

    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLUE)

    title = pyfiglet.figlet_format("Aloy Game Launcher", font="small")
    title_window = curses.newwin(10, 100, 1, 2)
    title_window.addstr(0, 0, title, curses.color_pair(1))
    title_window.refresh()
    stdscr.refresh()

    stdscr.addstr(6, 5, "Available Games:", curses.A_BOLD)
    for i, (game_name, game_reg) in enumerate(__registered_games__.items()):
        if i == 0:
            stdscr.addstr(
                8 + i, 7,
                f"{game_name}: {game_reg.description}",
                curses.color_pair(2)
            )
        else:
            stdscr.addstr(
                8 + i, 7, 
                f"{game_name}: {game_reg.description}",
                curses.A_NORMAL
            )
    title_window.refresh()
    stdscr.refresh()

    index: int = 0
    game_items_list = list(__registered_games__.items())
    char = stdscr.getch()
    title_window.refresh()
    while True:
        previous_index = index
        if char == ord("q"):
            return None
        elif char == curses.KEY_UP:
            index = max(index - 1, 0)
        elif char == curses.KEY_DOWN:
            index = min(index + 1, len(__registered_games__) - 1)
        elif char == curses.KEY_ENTER or char == 10 or char == 13:
            break
        if index != previous_index:
            game_name, game_reg = game_items_list[previous_index]
            stdscr.addstr(
                8 + previous_index, 7,
                f"{game_name}: {game_reg.description}",
                curses.A_NORMAL
            )
            game_name, game_reg = game_items_list[index]
            stdscr.addstr(
                8 + index, 7,
                f"{game_name}: {game_reg.description}",
                curses.color_pair(2)
            )
        char = stdscr.getch()
        title_window.refresh()

    stdscr.clear()
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.curs_set(True)
    curses.endwin()
    return list(__registered_games__.keys())[index]


def main() -> int:
    """Run a game on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--game",
        type=str,
        choices=list(__registered_games__.keys()),
        help="The name of the game to run."
    )
    parser.add_argument(
        "-l",
        "--list-games",
        action="store_true",
        help="List the available games."
    )
    parser.add_argument(
        "--launcher",
        action="store_true",
        help="Run the game launcher."
    )
    add_standard_parameters(parser)
    args: argparse.Namespace = parser.parse_args()

    if args.list_games:
        print("Available games:")
        for _name, _game in __registered_games__.items():
            print(f"\t{_name}: {_game.description}")
            print(f"\t\tDefault size: {_game.default_size}")
            print(f"\t\tCompetitive: {_game.competitive}")
            print(f"\t\tParameters:")
            for parameter in _game.parameters:
                print(f"\t\t\t{parameter}")
        return 0

    if args.launcher:
        print("Launching game launcher...")
        game = curses.wrapper(run_game_launcher)
        if game is None:
            return 0
    else:
        game = args.game

    if game not in __registered_games__:
        print(f"Game '{game}' is not registered.")
        return 1

    print(f"Launching game '{game}'...")
    game_registration: GameRegistration = __registered_games__[game]

    width = args.width
    height = args.height
    if width is None:
        width = game_registration.default_size[0]
    if height is None:
        height = game_registration.default_size[1]

    final_params: dict[str, Any] = {}
    for parameter in game_registration.parameters:
        for param_name in parameter.names:
            if param_name in args.params:
                final_params[param_name] = args.params[param_name]
                break
        else:
            final_params[param_name] = parameter.default

    importlib.import_module(
        f".{game_registration.module}",
        package=game_registration.package
    )
    entry_point = getattr(
        sys.modules[f"{game_registration.package}.{game_registration.module}"],
        game_registration.entry_point
    )

    return entry_point(
        width,
        height,
        args.debug,
        **final_params
    )


if __name__ == "__main__":
    sys.exit(main())
