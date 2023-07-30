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

"""Module for making plots in the terminal using ASCII characters."""

import enum
import math
from typing import Final, Literal, Sequence
import warnings
import numpy as np

from aloy.auxiliary.stringutils import StringBuilder

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "TerminalPlotColours",
    "make_terminal_relplot",
    "make_terminal_hisplot"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class TerminalPlotColours(enum.Enum):
    """Enum for the colours used in the terminal plots."""

    RED = "\033[91m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"


__EXTRA_COLOUR_LEN: Final[int] = 5
__RESET_COLOUR: Final[str] = "\033[0m"


def make_terminal_relplot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    plot_width: int = 80,
    plot_height: int = 40, /,
    title: str | None = None,
    legend: Sequence[str] | None = None,
    markers: Sequence[str] = ("x", "o", "+"),
    colours: Sequence[TerminalPlotColours] = (
        TerminalPlotColours.RED,
        TerminalPlotColours.BLUE,
        TerminalPlotColours.GREEN
    ),
    fill: bool = False,
    x_min: float = -np.inf,
    x_max: float = np.inf,
    y_min: float = -np.inf,
    y_max: float = np.inf
) -> str:
    """
    Make a relational line plot in the terminal using ASCII characters.

    Data must be given as 1-D or 2-D numpy arrays of the same shapes,
    and is normalised to fit the given plot width and height.
    If the array is 2-D, each row is plotted as a separate line.

    Parameters
    ----------
    `x_data: np.ndarray` - The x-axis data to plot.

    `y_data: np.ndarray` - The y-axis data to plot.

    `plot_width: int = 80` - The width of the plot in characters.

    `plot_height: int = 20` - The height of the plot in characters.

    `title: str | None = None` - The title of the plot. If None, no title is
    added.

    `legend: Sequence[str] | None = None` - The legend of the plot. If None,
    no legend is added.

    `markers: Sequence[str] = ("x", "o", "+")` - The markers to use for each
    set of data points. If there are more sets of data points than markers,
    the markers are cycled.

    `colours: Sequence[TerminalPlotColours] = (TerminalPlotColours.RED,
    TerminalPlotColours.GREEN, TerminalPlotColours.BLUE)` - The colours to
    use for each set of data points. If there are more sets of data points
    than colours, the colours are cycled.

    `fill: bool = False` - Whether to fill the area under the lines.

    `x_min: np.number = -np.inf` - The minimum value of the x-axis.

    `x_max: np.number = np.inf` - The maximum value of the x-axis.

    `y_min: np.number = -np.inf` - The minimum value of the y-axis.

    `y_max: np.number = np.inf` - The maximum value of the y-axis.

    Returns
    -------
    `str` - The relational line plot as a string.
    """
    if x_data.shape != y_data.shape:
        raise ValueError("x_data and y_data must be the same shape")
    if x_data.ndim > 2:
        raise ValueError("x_data and y_data must be 1-D or 2-D")
    if x_data.shape[-1] < plot_width:
        warnings.warn(
            "x_data has fewer elements than the plot width. "
            "Plot will be sparse."
        )
    if y_data.shape[-1] < plot_height:
        warnings.warn(
            "y_data has fewer elements than the plot height. "
            "Plot will be sparse."
        )

    # Clip the data to the given limits
    x_data = np.clip(x_data, x_min, x_max)
    y_data = np.clip(y_data, y_min, y_max)

    # Normalise the data to fit the plot
    x_interval = x_data.max() - x_data.min()
    y_interval = y_data.max() - y_data.min()
    normalised_x_data = (((x_data - x_data.min())
                          * (plot_width - 1))
                         / x_interval).astype(int)
    normalised_y_data = (((y_data - y_data.min())
                          * (plot_height - 1))
                         / y_interval).astype(int)

    # Create the plot grid, and fill it with the data points
    grid = np.full((plot_height, plot_width), fill_value=" ", dtype=object)
    if normalised_x_data.ndim == 1:
        update_grid(
            normalised_x_data,
            normalised_y_data,
            grid,
            markers[0],
            colours[0],
            fill
        )
    else:
        for i in range(normalised_x_data.shape[0]):
            update_grid(
                normalised_x_data[i],
                normalised_y_data[i],
                grid,
                markers[i % len(markers)],
                colours[i % len(colours)],
                fill
            )

    # Calculate the left-padding for the y-axis ticks and the x-axis tick gap
    y_tick_padding = len(f"{y_data.max():.2f}")
    x_tick_gap = len(f"{x_data.max():.2f}") + 1

    # Create the string builder and add the title and legend
    string_builder = StringBuilder()
    add_title(
        x_data,
        plot_width,
        title,
        legend,
        markers,
        colours,
        y_tick_padding,
        string_builder
    )

    # Add the y-axis, ticks and labels, and the plot itself
    for index, row in zip(range(plot_height, 0, -1), reversed(grid)):
        y_tick = (y_interval * (index / (plot_height - 1))) + y_data.min()
        string_builder += f" {y_tick:>{y_tick_padding}.2f} | "
        string_builder.extend(row, sep=" ", end="\n")

    def make_x_tick(index: int) -> str:
        return (x_interval * (index / (plot_width - 1))) + x_data.min()

    # Add the x-axis, ticks and labels
    string_builder += " " * (y_tick_padding + 2)
    string_builder += "=" * ((plot_width * 2) + 2)
    string_builder += "\n"
    string_builder += " " * (y_tick_padding + 4)
    string_builder.extend(
        "^" * math.floor((plot_width * 2) / (x_tick_gap + 1)),
        sep=(" " * x_tick_gap),
        end="\n"
    )
    string_builder += " " * (y_tick_padding + 4)
    string_builder.extend(
        f"{make_x_tick(index):<{x_tick_gap + 1}.2f}"
        for index in range(0, plot_width * 2, (x_tick_gap + 1))
        if index + x_tick_gap + 1 <= plot_width * 2
    )

    return string_builder.compile()


def make_terminal_hisplot(
    data: np.ndarray,
    plot_width: int = 80,
    plot_height: int = 40, /,
    title: str | None = None,
    legend: Sequence[str] | None = None,
    markers: Sequence[str] = ("x", "o", "+"),
    colours: Sequence[TerminalPlotColours] = (
        TerminalPlotColours.RED,
        TerminalPlotColours.GREEN,
        TerminalPlotColours.BLUE,
    ),
    kind: Literal["count", "percent"] = "count",
    data_min: float = -np.inf,
    data_max: float = np.inf
) -> str:
    """
    Make a histogram density plot in the terminal using ASCII characters.

    Data must be given as 1-D or 2-D numpy arrays, and is cut and normalised
    to fit the given plot width and height. If data is 2-D, each row is
    treated as a separate set of data points. The histogram of each set of
    data points is then stacked on top of each other.

    Parameters
    ----------
    `data: np.ndarray[ndim=1 or 2]` - The data to plot. If 2-D, each row is
    treated as a separate set of data points.

    `plot_width: int = 80` - The width of the plot in characters.

    `plot_height: int = 40` - The height of the plot in characters.

    `title: str | None = None` - The title of the plot. If None, no title is
    added.

    `legend: Sequence[str] | None = None` - The legend of the plot. If None,
    no legend is added.

    `markers: Sequence[str] = ("x", "o", "+")` - The markers to use for each
    set of data points. If there are more sets of data points than markers,
    the markers are cycled.

    `colours: Sequence[TerminalPlotColours] = (TerminalPlotColours.RED,
    TerminalPlotColours.GREEN, TerminalPlotColours.BLUE)` - The colours to
    use for each set of data points. If there are more sets of data points
    than colours, the colours are cycled.

    `kind: Literal["count", "percent"] = "count"` - Whether to plot the
    counts or the percentage of each bin.

    `data_min: np.number = -np.inf` - The minimum value of the data to plot.

    `data_max: np.number = np.inf` - The maximum value of the data to plot.

    Returns
    -------
    `str` - The histogram density plot as a string.
    """
    if data.shape[-1] < plot_height:
        warnings.warn(
            "data has fewer elements than the plot height. "
            "Plot will be sparse."
        )

    # Calculate the bin edges
    if data.ndim == 1:
        data = data[np.newaxis, :]
    bin_edges = np.linspace(data.min(), data.max(), plot_height + 1)

    # Calculate the bin counts (this is the x-axis data)
    bin_counts = np.zeros((data.shape[0], plot_height))
    for i in range(data.shape[0]):
        bin_counts[i, :] = np.histogram(
            data[i], bins=bin_edges, range=(data_min, data_max)
        )[0]
    if kind == "percent":
        bin_counts = ((bin_counts / bin_counts.sum()) * 100).astype(int)
    count_min = bin_counts.nonzero()[0].min()  # pylint: disable=no-member
    count_max = bin_counts.max(axis=1).sum()
    bin_counts = ((bin_counts * (plot_width - 1)) / count_max).astype(int)

    # Calculate the bin centres (this is the y-axis data)
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Calculate the left-padding for the y-axis ticks and the x-axis tick gap
    y_tick_padding = max(
        len(f"{bin_counts.min():.2f}"),
        len(f"{bin_counts.max():.2f}")
    )
    x_tick_gap = len(f"{count_max:.2f}") + 1

    # Create the string builder and add the title and legend
    string_builder = StringBuilder()
    add_title(
        data,
        plot_width,
        title,
        legend,
        markers,
        colours,
        y_tick_padding,
        string_builder
    )

    # Add the y-axis, ticks and labels, and the plot itself
    for y, i in zip(bin_centres, range(bin_counts.shape[1])):
        string_builder += f" {y:>{y_tick_padding}.2f} | "
        for j, x in enumerate(bin_counts[:, i]):
            if x != 0:
                string_builder.append_many(
                    f"{colours[j % len(colours)].value}"
                    f"{markers[j % len(markers)]}{__RESET_COLOUR}",
                    x,
                    sep=" "
                )
                if j != bin_counts.shape[1] - 1:
                    string_builder += " "
        string_builder += "\n"

    def make_x_tick(index: int) -> str:
        return (((count_max - count_min)
                 * (index / (plot_width - 1)))
                + count_min)

    # Add the x-axis, ticks and labels
    string_builder += " " * (y_tick_padding + 2)
    string_builder += "=" * ((plot_width * 2) + 2)
    string_builder += "\n"
    string_builder += " " * (y_tick_padding + 4)
    string_builder.extend(
        "^" * math.floor((plot_width * 2) / (x_tick_gap + 1)),
        sep=(" " * x_tick_gap),
        end="\n"
    )
    string_builder += " " * (y_tick_padding + 4)
    string_builder.extend(
        f"{make_x_tick(index):<{x_tick_gap + 1}.2f}"
        for index in range(0, plot_width * 2, (x_tick_gap + 1))
        if index + x_tick_gap + 1 <= plot_width * 2
    )

    return string_builder.compile()


def update_grid(
    normalised_x_data: np.ndarray,
    normalised_y_data: np.ndarray,
    grid: np.ndarray,
    marker: str,
    color: TerminalPlotColours,
    fill: bool
) -> None:
    """Update the grid with the normalised data points."""
    marker = color.value + marker + __RESET_COLOUR
    if not fill:
        grid[normalised_y_data, normalised_x_data] = marker
    else:
        index_range = np.arange(normalised_y_data.shape[0])
        y_indices = index_range <= normalised_y_data[:, np.newaxis]
        for y, x in zip(y_indices, normalised_x_data):
            grid[index_range[y], x] = marker


def add_title(
    x_data: np.ndarray,
    plot_width: int,
    title: str | None,
    legend: Sequence[str] | None,
    markers: Sequence[str],
    colours: Sequence[TerminalPlotColours],
    y_tick_padding: int,
    string_builder: StringBuilder
) -> None:
    """Add the title and legend to the string builder."""
    if title is not None:
        string_builder.set_duplicator_flag("title frame start")
        string_builder += " " * (y_tick_padding + 4)
        string_builder += ("-" * (len(title) + 2)).center(plot_width * 2)
        string_builder.set_duplicator_flag("title frame end")
        string_builder += "\n"
        string_builder += " " * (y_tick_padding + 4)
        string_builder += title.center(plot_width * 2)
        string_builder += "\n"
        string_builder.duplicate_flagged(
            "title frame start",
            "title frame end"
        )
    if legend is not None:
        if len(legend) != x_data.shape[0]:
            raise ValueError(
                "Legend must have the same number of elements as "
                f"data has rows. Got; legend = {len(legend)} "
                f"and data rows = {x_data.shape[0]}."
            )
        if title is not None:
            string_builder += "\n"
        string_builder += " " * (y_tick_padding + 4)
        string_builder += (
            " || ".join(
                ((f"[{colours[i % len(colours)].value}"
                  f"{markers[i % len(markers)]}{__RESET_COLOUR}] {label}")
                 for i, label in enumerate(legend))
            )
        ).center(((plot_width + (len(legend) * __EXTRA_COLOUR_LEN)) * 2) - 2)
        string_builder += "\n"
    elif title is not None:
        string_builder += "\n"


if __name__ == "__main__":
    print(
        make_terminal_relplot(
            np.tile(np.arange(0, 200), 3).reshape(3, 200),
            np.array(
                [
                    (np.linspace(0, 40, 200) ** 2),
                    (np.linspace(0, 40, 200) ** 1.5),
                    np.linspace(0, 40, 200)
                ]
            ),
            40,
            20,
            title="Jake is the best <3",
            legend=["y = x^2", "y = x^1.5", "y = x"]
        )
    )
    print(
        make_terminal_hisplot(
            np.random.normal(0, 1, 1000),
            40,
            20,
            title="Normal Distribution",
            legend=["y = N(0, 1)"],
            kind="count",
            data_min=-5,
            data_max=5
        ),
        end="\n"
    )
    print(
        make_terminal_hisplot(
            np.array(
                [
                    np.random.normal(0, 1, 1000),
                    np.random.normal(0, 2.5, 1000)
                ]
            ),
            40,
            20,
            title="Normal Distribution",
            legend=["y = N(0, 1)", "y = N(0, 2.5)"],
            kind="percent",
            data_min=-5,
            data_max=5
        ),
        end="\n"
    )
