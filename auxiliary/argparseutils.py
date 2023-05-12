
import argparse
from typing import Iterable, Mapping, Optional, Sequence


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
        option_string: Optional[str] = None
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
