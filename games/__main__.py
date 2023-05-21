"""Run a game on the command line."""

import argparse
import importlib
import importlib.util
import sys
from typing import Any, Callable, Final, Iterable, NamedTuple

from auxiliary.argparseutils import mapping_argument_factory
from games.snakegame import play_snake_game


class GameParam(NamedTuple):
    """A class to store information about a game parameter."""

    names: list[str]
    type_: type
    default: Any
    help: str


__STANDARD_PARAMS: Final[list[GameParam]] = [
    GameParam(["-w", "--width"], int, 1200, "The width of the game window."),
    GameParam(["-t", "--height"], int, 800, "The height of the game window."),
    GameParam(["--debug"], bool, False, "Run the game in debug mode.")
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
    """A class to store information about a game."""

    entry_point: Callable
    module: str
    name: str
    description: str
    parameters: list[GameParam]
    package: str = "games"


__registered_games__: dict[str, GameRegistration] = {}


def register_game(
    entry_point: Callable,
    module: str,
    name: str,
    description: str,
    parameters: Iterable[GameParam] | None = None,
    package: str = "games"
) -> None:
    """
    Register a game with the game launcher.

    Parameters
    ----------
    `entry_point: Callable` - The function to call to run the game.

    `module: str` - The name of the module containing the game.

    `name: str` - The name of the game.

    `description: str` - A description of the game. This will be displayed
    when the user lists the available games.

    `parameters: dict[str, GameParam] | None = None` - A dictionary of
    parameters for the game. The keys are the names of the parameters, and the
    values are `GameParam` objects, containing the type, default value, and
    help text for the parameter. The keys must be valid Python identifiers,
    and must not be the same as any of the standard parameters, see
    `jinx.games.standard_params()`. Note that the parameters will be passed
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
        parameters=parameters,
        package=package
    )


register_game(
    play_snake_game,
    "snakegame",
    "snake",
    "Play the snake game."
)


def main() -> int:
    """Run a game on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--game",
        type=str,
        choices=["snake", "tetris", "pacman"],
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
        return 0

    if args.launcher:
        print("Launching game launcher...")
        print("Not implemented yet...")  # TODO: Implement game launcher.
        # game = run_game_launcher()
        return 0
    else:
        game: str = args.game

    if game not in __registered_games__:
        print(f"Game '{game}' is not registered.")
        return 1

    print(f"Launching game '{game}'...")
    game_registration: GameRegistration = __registered_games__[game]

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

    return game_registration.entry_point(
        args.width,
        args.height,
        args.debug,
        **final_params
    )


if __name__ == "__main__":
    sys.exit(main())
