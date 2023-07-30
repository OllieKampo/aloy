

import datetime
import json
import os
import random
from typing import Generic, NamedTuple, TypeAlias, TypeVar

ValidJSONPrimitive: TypeAlias = str | int | float | bool | None
ValidJSONType: TypeAlias = (
    ValidJSONPrimitive  # type: ignore
    | list[ValidJSONPrimitive]
    | dict[str, ValidJSONPrimitive]
)


class GameSpec(NamedTuple):
    """Class storing options for a game specification."""

    game_name: str
    match_name: str
    game_options: ValidJSONType | list[ValidJSONType] | dict[str, ValidJSONType]


AT = TypeVar("AT")
ST = TypeVar("ST")


class GameRecorder(Generic[AT, ST]):  # pylint: disable=too-few-public-methods
    """Class recording actions played in a game, and saving them to a file."""

    def __init__(
        self,
        game_spec: GameSpec,
        initial_state: ST | None = None
    ) -> None:
        """Initialize a GameRecorder."""
        self.__game_spec: list[GameSpec] = [game_spec]
        self.__initial_state: list[ST | None] = [initial_state]
        self.__records: list[list[dict[str, AT | ST | None]]] = [[]]
        self.__last_game_spec: GameSpec = game_spec
        self.__last_initial_state: ST | None = initial_state

    def new_match(
        self,
        game_spec: GameSpec | None = None,
        initial_state: ST | None = None
    ) -> None:
        """Start recording a new match."""
        if game_spec is None:
            if self.__game_spec:
                game_spec = self.__game_spec[-1]
            else:
                game_spec = self.__last_game_spec
        self.__last_game_spec = game_spec
        self.__game_spec.append(game_spec)
        if initial_state is None:
            if self.__initial_state:
                initial_state = self.__initial_state[-1]
            else:
                initial_state = self.__last_initial_state
        self.__last_initial_state = initial_state
        self.__initial_state.append(initial_state)
        self.__records.append([])

    def clear_last_match(self) -> None:
        """Clear the last match."""
        if self.__game_spec:
            self.__game_spec.pop()
            self.__initial_state.pop()
            self.__records.pop()

    def record(self, action: AT, state: ST | None = None) -> None:
        """Record an action."""
        self.__records[-1].append({"action": action, "state": state})

    def save(self, filename: str, append: bool = True) -> None:
        """Save the recorded actions to a file."""
        data = [
            {
                "game": self.__game_spec[i].game_name,
                "match": self.__game_spec[i].match_name,
                "id": i,
                "date": datetime.datetime.now().isoformat(),
                "options": self.__game_spec[i].game_options,
                "initial_state": self.__initial_state[i],
                "records": self.__records[i]
            }
            for i in range(len(self.__game_spec))
        ]
        # For open modes see https://stackoverflow.com/a/30566011
        # For IO see https://docs.python.org/3/library/io.html#i-o-base-classes
        # Since UTF-8 does not have uniform character width (variable-length
        # character encoding), we cannot use seek() reliably move a number of
        # characters relative to the end of the file. See here for discussion:
        #   - https://stackoverflow.com/a/53310360
        #   - https://stackoverflow.com/a/18857381
        #   - https://stackoverflow.com/a/21533561
        #   - https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename) and os.path.isfile(filename) and append:
            with open(filename, "ab+") as file:
                file.seek(0, os.SEEK_END)
                if (tell := file.tell()) == 0:
                    file.write("[\n".encode())
                else:
                    file.seek(tell - len("\n]".encode()), os.SEEK_SET)
                    file.truncate()
                    file.write(",\n".encode())
                file.write(json.dumps(data, indent=2)[2:].encode())
        else:
            with open(filename, "w") as file:
                json.dump(data, file, indent=2)
        self.__game_spec = []
        self.__initial_state = []
        self.__records = []


def __main() -> None:
    ops = GameSpec("test_game", "match_1", {"difficulty": "hard"})
    rec = GameRecorder[str, list[int]](ops, [1, 2, 3, 4])
    rec.record("forwards", [2, 2, 3, 4])
    rec.save("test_game_save")

    # Read the file back in and print it
    with open("test_game_save", "r") as file:
        print(json.load(file))


if __name__ == "__main__":
    __main()
