
"""This module contains helper classes and functions for data handling with pandas."""

from typing import Any, Iterable, Optional, Type
import pandas as pd

class DataHolder:
    """Holds data to be converted into a pandas dataframe."""
    
    __slots__ = ("__data", "__headers", "__converters") 
                ## Columns index. Could inclused provision for hierarchical index?
                ## List of lists (one for each row in the table), or dictionary of dictionaries
                ## with arbitrary depths, where the lowest depths are made into individual single
                ## index dataframes, and higher dictionary levels concatenate those dictionaries
                ## with the higher level header names as the multi-index keys.
    
    def __init__(self,
                 headers: list[str | tuple[str, ...]],
                 converters: Optional[dict[str, Type]] = None,
                 configuration: Optional[dict[str, Any]] = None,
                 configuration_as_index: bool = False
                 ) -> None:
        """
        Create a data holder for gathering and storing data generated during a algorithm or system trial.
        
        Parameters
        ----------
        `headers : list[str | tuple[str, ...]]` - The column headers for the data.

        `converters : Optional[dict[str, Type]]` - A dictionary mapping column headers to types to convert the data to.

        `configuration : Optional[dict[str, Any]]` - A dictionary containing the configuration of the algorithm or system being tested.

        `configuration_as_index : bool` - Whether to add the configuration as a multi-index to the dataframe's rows.
        Otherwise, they are added as columns.
        """
        self.__data = [[] for _ in headers]
        self.__headers: list[str | tuple[str, ...]] = headers
        self.__converters: dict[str, Type] = converters if converters is not None else {}
    
    def add_row(self,
                data: list[Any], /,
                upper_headers: Optional[list[str]] = None
                ) -> None:
        """Add a row of data under the given hierarchical column index headers."""
        if not len(data) == len(self.__headers):
            raise ValueError("Data length does not match headers length.")
        for index, (item, header) in enumerate(zip(data, self.__headers)):
            self.__data[index].append(item if header not in self.__converters
                                      else self.__converters[header](item))
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.__data, columns=self.__headers)


# def combine_trials(dataframes: Iterable[pd.DataFrame],
#                    on_rows: bool = True):
#     """Combines a set of dataframes representing different trials of an algorithm or system under different parameter configurations."""
#     pass


class MultiTableDataHolder:
    """
    Data holder that can handle multiple dataframes, each representing a different table of data for the same experimental trial of an algorithm or system under the same parameter configuration.
    """
    pass


class TrialDataCollector:
    """
    Wraps multiple data holders, and can handle multiple dataframes, each representing a different trial of an algorithm or system under different parameter configurations.
    """
    pass
