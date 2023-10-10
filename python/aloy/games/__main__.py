###############################################################################
# Copyright (C) 2023 Oliver Michael Kamperis
# Email: olliekampo@gmail.com
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Run a game on the command line."""

import argparse
import importlib
import importlib.util
import sys
from typing import Any, Callable, Final, Iterable, NamedTuple
import tomllib
import os

import curses
import pyfiglet

from aloy.auxiliary.argparseutils import (
    mapping_argument_factory,
    optional_bool,
    optional_int
)

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "1.0.0"

__all__ = ()


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class _GameParam(NamedTuple):
    """
    A class to store information about a game parameter.

    Items
    -----
    `names: list[str]` - A list of names for the parameter. The first name
    should be the shortest, and the last name should be the longest. The
    longest name will be used as the true name of the parameter.

    `type_: type | Callable[[str], Any]` - The type of the parameter. This
    can be a Python type, or a function that takes a string and returns the
    parameter value.

    `default: Any` - The default value of the parameter.

    `help: str` - The help text for the parameter.
    """

    names: list[str]
    type_: type | Callable[[str], Any]
    help: str
    default: Any
    const: Any | None = None
    nargs: int | str | None = None
    choices: Iterable[Any] | None = None


_STANDARD_PARAMS: Final[list[_GameParam]] = [
    _GameParam(
        names=["-l", "--list-games"],
        type_=optional_bool,
        help="List the available games.",
        default=False,
        const=True,
        nargs="?"
    ),
    _GameParam(
        names=["--launcher"],
        type_=optional_bool,
        help="Run the game launcher.",
        default=False,
        const=True,
        nargs="?"
    ),
    _GameParam(
        names=["-w", "--width"],
        type_=optional_int,
        help="The width of the game window, each game has its own default.",
        default=None
    ),
    _GameParam(
        names=["-t", "--height"],
        type_=optional_int,
        help="The height of the game window, each game has its own default.",
        default=None
    ),
    _GameParam(
        names=["--debug"],
        type_=optional_bool,
        help="Run the game in debug mode.",
        default=False,
        const=True,
        nargs="?"
    )
]


def _add_standard_parameters(parser: argparse.ArgumentParser) -> None:
    """Add standard game parameters to the argument parser."""
    for parameter in _STANDARD_PARAMS:
        parser.add_argument(
            *parameter.names,
            type=parameter.type_,
            help=parameter.help,
            default=parameter.default,
            const=parameter.const,
            nargs=parameter.nargs,
            choices=parameter.choices
        )
    parser.add_argument(
        "-params",
        nargs="*",
        action=mapping_argument_factory(),
        type=str,
        metavar="KEY_1=VALUE_1 KEY_i=VALUE_i [...] KEY_n=VALUE_n",
        help="Parameters for the game."
    )


_NAME_RESERVED_CHARACTERS: Final[set[str]] = {"-", "_", " "}


class _GameRegistration(NamedTuple):
    """
    Tuple to store information about a game registration.

    Items
    -----
    `module: str` - The name of the module containing the game.

    `entry_point: str` - The name of the function to call to run the game.

    `name: str` - The name of the game as it will be displayed to the user.

    `description: str` - A description of the game. This will be displayed
    when the user lists the available games.

    `default_size: tuple[int, int]` - The default size of the game window in
    pixels (width, height).

    `competitive: bool = False` - Whether the game is competitive. If the game
    is competitive, the game launcher will ask the user if they want to play
    against the computer or another player.

    `parameters: list[GameParam]` - A list of parameters for the game. The
    keys are the names of the parameters, and the values are `GameParam`
    objects, containing the type, default value, and help text for the
    parameter. The keys must be valid Python identifiers without underscores,
    and must not be the same as any of the standard parameters, see
    `aloy.games.standard_params()`. Note that the parameters will be passed to
    the game entry point as keyword arguments.
    """

    module: str
    entry_point: str
    name: str
    description: str
    default_size: tuple[int, int]
    competitive: bool
    parameters: list[_GameParam]
    package: str = "aloy.games"


_REGISTERED_GAMES: dict[str, _GameRegistration] = {}


def _register_game(
    module: str,
    entry_point: str,
    name: str,
    description: str,
    default_size: tuple[int, int],
    competitive: bool = False,
    parameters: Iterable[_GameParam] | None = None,
    package: str = "aloy.games"
) -> None:
    """
    Register a game with the game launcher.

    Parameters
    ----------
    `module: str` - The name of the module containing the game.

    `entry_point: str` - The name of the function to call to run the game.

    `name: str` - The name of the game as it will be displayed to the user.

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
    help text for the parameter. The keys must be valid Python identifiers
    without underscores, and must not be the same as any of the standard
    parameters, see `aloy.games.standard_params()`. Note that the parameters
    will be passed to the game entry point as keyword arguments.

    `package: str = "games"` - The name of the package containing the game
    module if it is not in the root package.
    """
    # Check if the module exists.
    spec = importlib.util.find_spec(f".{module}", package=package)
    if spec is None:
        raise ValueError(f"Module '{module}' does not exist.")

    # Check name is valid.
    if name in _REGISTERED_GAMES:
        raise ValueError(f"Game '{name}' is already registered.")
    for char in _NAME_RESERVED_CHARACTERS:
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
        parameter.names.sort(key=len)
        for _name in parameter.names:
            for standard_parameter in _STANDARD_PARAMS:
                if _name in standard_parameter.names:
                    raise ValueError(
                        f"Game parameter '{_name}' is a standard parameter."
                    )
    parameters = list(parameters)

    _REGISTERED_GAMES[name] = _GameRegistration(
        entry_point=entry_point,
        module=module,
        name=name,
        description=description,
        default_size=default_size,
        competitive=competitive,
        parameters=parameters,
        package=package
    )


def _run_game_launcher(stdscr: curses.window) -> str | None:
    """Run a game launcher with curses."""
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
    for i, (game_name, game_reg) in enumerate(_REGISTERED_GAMES.items()):
        if i == 0:
            stdscr.addstr(
                8 + i, 7,
                f"{game_name}: {game_reg.description}",
                curses.color_pair(2)
            )
        else:
            stdscr.addstr(
                8 + i,
                7,
                f"{game_name}: {game_reg.description}",
                curses.A_NORMAL
            )
    title_window.refresh()
    stdscr.refresh()

    index: int = 0
    game_items_list = list(_REGISTERED_GAMES.items())
    char = stdscr.getch()
    title_window.refresh()
    while True:
        previous_index = index
        if char == ord("q"):
            return None
        elif char == curses.KEY_UP:
            index = max(index - 1, 0)
        elif char == curses.KEY_DOWN:
            index = min(index + 1, len(_REGISTERED_GAMES) - 1)
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
    return list(_REGISTERED_GAMES.keys())[index]


_TYPE_MAP = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool
}


def _get_type(type_name: str) -> type:
    """Get a type from a string."""
    if type_name not in _TYPE_MAP:
        raise ValueError(f"Unknown type '{type_name}'.")
    return _TYPE_MAP[type_name]


def _register_all_games() -> None:
    """Register all games from the '_games.toml' file."""
    file_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_path, "_games.toml")
    with open(file_path, "rb") as file:
        game_dict = tomllib.load(file)

    for _, game in game_dict["games"]["register"].items():
        _register_game(
            module=game["module"],
            entry_point=game["entry_point"],
            name=game["name"],
            description=game["description"],
            default_size=game["default_size"],
            competitive=game["competitive"],
            parameters=[
                _GameParam(
                    names=[_param],
                    type_=_get_type(_spec["type"]),
                    default=_spec["default"],
                    help=_spec["help"]
                )
                for _param, _spec in game["parameters"].items()
            ]
        )


def _main() -> int:
    """Run a game on the command line."""
    # Register all games.
    _register_all_games()

    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--game",
        type=str,
        help="The name of the game to run.",
        choices=list(_REGISTERED_GAMES.keys())
    )
    _add_standard_parameters(parser)
    args: argparse.Namespace = parser.parse_args()

    # List the available games if requested.
    if args.list_games:
        print("Available games:")
        for _name, _game in _REGISTERED_GAMES.items():
            print(f"\t{_name}: {_game.description}")
            print(f"\t\tDefault size: {_game.default_size}")
            print(f"\t\tCompetitive: {_game.competitive}")
            print("\t\tParameters:")
            for parameter in _game.parameters:
                print(f"\t\t\t{parameter}")
        return 0

    # Get the game to play.
    if args.game is None and not args.launcher:
        print("No game specified.")
        return 1
    if args.launcher:
        print("Launching game launcher...")
        game = curses.wrapper(_run_game_launcher)
        if game is None:
            return 0
    else:
        game = args.game
    if game not in _REGISTERED_GAMES:
        print(f"Game '{game}' is not registered.")
        return 1
    print(f"Launching game '{game}'...")
    game_registration: _GameRegistration = _REGISTERED_GAMES[game]

    # Get window size.
    width = args.width
    height = args.height
    if width is None:
        width = game_registration.default_size[0]
    if height is None:
        height = game_registration.default_size[1]

    # Get parameters.
    final_params: dict[str, Any] = {}
    for parameter in game_registration.parameters:
        true_name = parameter.names[-1]
        if args.params is not None:
            for param_name in parameter.names:
                if param_name in args.params:
                    final_params[true_name] = args.params[param_name]
                    break
        if true_name not in final_params:
            final_params[true_name] = parameter.default

    # Import the module and get the entry point.
    importlib.import_module(
        f".{game_registration.module}",
        package=game_registration.package
    )
    entry_point = getattr(
        sys.modules[f"{game_registration.package}.{game_registration.module}"],
        game_registration.entry_point
    )

    # Run the game.
    return entry_point(
        width=width,
        height=height,
        debug=args.debug,
        **final_params
    )


if __name__ == "__main__":
    sys.exit(_main())
