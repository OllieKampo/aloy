# Copyright (C) 2023 Oliver Michael Kamperis
# Email: o.m.kamperis@gmail.com
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Module defining argument parsing utilities."""

import argparse
from numbers import Number
from typing import Any, Iterable, Mapping, Sequence


def bool_options(
    default: bool | None = None,
    const: bool | None = True,
    add_none: bool = False
) -> dict[str, Any]:
    """
    Create a Boolean argument.

    Parameters
    ----------
    `default: bool | None = None` - The default argument value used when the
    argument is not given.

    `const: bool | None = True` - The standard argument value used when the
    argument is given without a value.

    `add_none: bool = False` - Whether to allow None to be valid arguments
    value.

    Returns
    -------
    `dict[str, Any]` - A dictionary of options for creating a Boolean argument.
    """
    choices: list[bool | None] = [True, False]
    if add_none:
        choices.append(None)
    return {
        "nargs": "?",
        "choices": choices,
        "default": default,
        "const": const,
        "type": optional_bool
    }


def optional_number(value: str) -> Number | None:
    """
    Optional number argument type.

    Return None if the value is an empty string or the string "None", otherwise
    return the input string parsed as a float if it contains a decimal point or
    otherwise as an integer.
    """
    if not value or value == "None":
        return None
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError as error:
        print(f"Cannot parse {value} as a float or int: {error}")
        raise error


def optional_int(value: str) -> int | None:
    """
    Optional integer argument type.

    Return None if the value is an empty string or the string "None", otherwise
    return the input string parsed as an integer.
    """
    if not value or value == "None":
        return None
    try:
        return int(value)
    except ValueError as error:
        print(f"Cannot parse {value} as int: {error}")
        raise error


def optional_str(value: str) -> str | None:
    """
    Optional string argument type.

    Return None if the value is an empty string or the string "None",
    otherwise return the input string.
    """
    if not value or value == "None":
        return None
    return value


def optional_bool(value: str) -> bool | None:
    """
    Optional boolean argument type.

    Return None if the value is an empty string or the string "None",
    otherwise return the input string parsed as a boolean.
    """
    if not value or value == "None":
        return None
    if value.lower() in ["true", "yes", "on"]:
        return True
    if value.lower() in ["false", "no", "off"]:
        return False
    message = f"Cannot parse {value} as a boolean."
    print(message)
    raise ValueError(message)


def mapping_argument_factory(
    choices: Iterable[str] | Mapping[str, Iterable[str]] | None = None,
    multi_values: bool = True,
    comma_replacer: str = "~"
) -> type[argparse.Action]:
    """
    Construct a special action for storing arguments of parameters given as a
    mapping.

    Parameters
    ----------
    `choices: Iterable[str] | Mapping[str, Iterable[str]] | None` - The
    allowed choices for the values of the mapping. If `None`, no checks are
    performed. If an `Iterable`, the items are the allowed keys. If a
    `Mapping`, the keys are the allowed keys and the values are the allowed
    values for the corresponding keys.

    `multi_values: bool` - Whether multiple values are allowed for a single
    key.

    `comma_replacer: str` - The string to replace commas with in the values of
    the mapping.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Sequence[str],
        option_string: str | None = None
    ) -> None:
        """
        Store the given values as a mapping in the namespace.

        Parameters
        ----------
        `parser: argparse.ArgumentParser` - The argument parser.

        `namespace: argparse.Namespace` - The namespace to store the values in.

        `values: Sequence[str]` - The values to parse.

        `option_string: str | None` - The option string that was used to
        invoke the action.
        """

        check_values: bool = False
        if isinstance(self.__class__.choices, Mapping):
            check_values = True

        _values: dict[str, str | list[str]] = {}
        try:
            for key_value in values:
                key, value = key_value.split('=', 1)
                if (self.__class__.choices is not None
                        and key not in self.__class__.choices):
                    _keys = list(self.__class__.choices.keys())
                    error_string = (
                        f"Error parsing mapping arg '{option_string}' "
                        f"for key-value mapping {key_value}. "
                        f"The key {key} is not allowed. "
                        f"Allowed keys are: {_keys}."
                    )
                    print(error_string)
                    raise RuntimeError(error_string)

                if ',' in value:
                    values = value
                    if not self.__class__.multi_values:
                        error_string = (
                            f"Error parsing mapping arg '{option_string}' "
                            f"for key-value mapping {key_value}. "
                            "Multiple values are not allowed."
                        )
                        print(error_string)
                        raise RuntimeError(error_string)
                    _values[key] = [
                        value.replace(self.__class__.comma_replacer, ',')
                        for value in values.split(',')
                    ]
                else:
                    value = value.replace(self.__class__.comma_replacer, ',')
                    if self.__class__.multi_values:
                        _values[key] = [value]
                    else:
                        _values[key] = value

                if check_values:
                    if self.__class__.multi_values:
                        values = _values[key]
                    else:
                        values = [_values[key]]
                    for value in _values[key]:
                        if value not in self.__class__.choices[key]:
                            _values = list(self.__class__.choices[key])
                            error_string = (
                                f"Error parsing mapping arg '{option_string}' "
                                f"for key-value mapping {key_value}. "
                                f"The value {value} is not allowed. "
                                f"Allowed values are: {_values}."
                            )
                            print(error_string)
                            raise RuntimeError(error_string)

        except ValueError as error:
            print(f"Error during parsing mapping argument '{option_string}' "
                  f"for key-value mapping '{key_value}': {error}.")
            raise error
        setattr(namespace, self.dest, _values)

    return type(
        "StoreMappingArgument",
        (argparse.Action,),
        {
            "__call__": __call__,
            "choices": list(choices) if choices is not None else None,
            "multi_values": multi_values,
            "comma_replacer": comma_replacer
        }
    )
