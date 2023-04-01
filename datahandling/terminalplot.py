
import math
from typing import Sequence
import warnings
import numpy as np

from auxiliary.stringutils import StringBuilder


def make_terminal_relplot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    plot_width: int,
    plot_height: int, /,
    title: str | None = None,
    fill: bool = False,
    markers: Sequence[str] = ("x", "o", "+"),
    legend: Sequence[str] | None = None,
    x_min: np.number = -np.inf,
    x_max: np.number = np.inf,
    y_min: np.number = -np.inf,
    y_max: np.number = np.inf
) -> str:
    """
    Make a simple relational line plot in the terminal using ASCII characters.

    Data must be given as 1-D or 2-D numpy arrays of the same shapes,
    and is normalised to fit the given plot width and height.
    If the array is 2-D, each row is plotted as a separate line.
    """
    if x_data.shape != y_data.shape:
        raise ValueError("x_data and y_data must be the same shape")
    if x_data.ndim > 2:
        raise ValueError("x_data and y_data must be 1-D or 2-D")
    if x_data.shape[-1] < plot_width:
        warnings.warn("x_data has fewer elements than the plot width. "
                      "Plot will be sparse.")
    if y_data.shape[-1] < plot_height:
        warnings.warn("y_data has fewer elements than the plot height. "
                      "Plot will be sparse.")

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
            fill
        )
    else:
        for i in range(normalised_x_data.shape[0]):
            update_grid(
                normalised_x_data[i],
                normalised_y_data[i],
                grid,
                markers[i % len(markers)],
                fill
            )

    # Calculate the left-padding for the y-axis ticks and the x-axis tick gap
    y_tick_padding = len(f"{y_data.max():.2f}")
    x_tick_gap = len(f"{x_data.max():.2f}") + 1

    # Create the string builder and add the title and legend
    string_builder = StringBuilder()
    if title is not None:
        string_builder += " " * (y_tick_padding + 4)
        string_builder += ("-" * (len(title) + 2)).center(plot_width * 2)
        string_builder += "\n"
        string_builder += " " * (y_tick_padding + 4)
        string_builder += title.center(plot_width * 2)
        string_builder += "\n"
        string_builder.duplicate(7, 4)
    if legend is not None:
        if len(legend) != x_data.shape[0]:
            raise ValueError("Legend must have the same number of elements as "
                             f"x_data has rows. Got; legend = {len(legend)} "
                             f"and x_data rows = {x_data.shape[0]}.")
        string_builder += "\n"
        string_builder += " " * (y_tick_padding + 4)
        string_builder += (" || ".join(
                (f"[{markers[i % len(markers)]}] {label}"
                 for i, label in enumerate(legend))
            )
        ).center(plot_width * 2)
        string_builder += "\n"
    elif title is not None:
        string_builder += "\n"

    # Add the y-axis, ticks and labels, and the plot itself
    for index, row in zip(range(plot_height, 0, -1), reversed(grid)):
        y_tick = (y_interval * (index / (plot_height - 1))) + y_data.min()
        string_builder += f" {y_tick:<{y_tick_padding}.2f} | "
        string_builder.extend(row, sep=" ", end="\n")

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
    def make_x_tick(index: int) -> str:
        return (x_interval * (index / (plot_width - 1))) + x_data.min()
    string_builder.extend(
        f"{make_x_tick(index):<{x_tick_gap + 1}.2f}"
        for index in range(0, plot_width, (x_tick_gap + 1) // 2)
    )

    return string_builder.compile()


def update_grid(
    normalised_x_data: np.ndarray,
    normalised_y_data: np.ndarray,
    grid: np.ndarray,
    marker: str,
    fill: bool
) -> None:
    """Update the grid with the normalised data points."""
    if not fill:
        grid[normalised_y_data, normalised_x_data] = marker
    else:
        index_range = np.arange(normalised_y_data.shape[0])
        y_indices = index_range <= normalised_y_data[:, np.newaxis]
        for y, x in zip(y_indices, normalised_x_data):
            grid[index_range[y], x] = marker


if __name__ == "__main__":
    print(make_terminal_relplot(np.arange(0, 200), np.linspace(1, 40, 200) ** 1.5, 40, 20, "Hello World"), end="\n")
    print(make_terminal_relplot(np.arange(0, 200), np.linspace(1, 40, 200) ** 1.5, 40, 20, "Hello World", True), end="\n")
    print(make_terminal_relplot(np.tile(np.arange(0, 200), 3).reshape(3, 200),
                                np.array([np.linspace(0, 40, 200), (np.linspace(0, 40, 200) ** 1.5), (np.linspace(0, 40, 200) ** 2)]),
                                40, 20, "Jake is the best <3", legend=["y = x * 5", "y = (x * 5) ^ 1.5", "y = (x * 5) ^ 2"]))
