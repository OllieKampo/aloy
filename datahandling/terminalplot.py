
import math
import warnings
import numpy as np

from auxiliary.stringutils import StringBuilder

def make_terminal_relplot(x_data: np.ndarray,
                          y_data: np.ndarray,
                          plot_width: int,
                          plot_height: int,
                          title: str,
                          markers: list[str] = ["x", "o", "+"]
                          ) -> str:
    """
    Make a simple relational line plot in the terminal using ASCII characters.

    Data must be given as 1-D or 2-D numpy arrays of the same shapes,
    and is normalised to fit the given plot width and height.
    If the array is 2-D, each row is plotted as a separate line,
    and the array can have at most 3 columns.
    """
    if x_data.shape != y_data.shape:
        raise ValueError("x_data and y_data must be the same shape")
    if x_data.shape[0] < plot_width:
        warnings.warn("x_data has fewer elements than the plot width. Plot will be sparse.")
    if y_data.shape[0] < plot_height:
        warnings.warn("y_data has fewer elements than the plot height. Plot will be sparse.")
    string_builder = StringBuilder(title.center(plot_width * 2))
    string_builder.append("\n")
    string_builder.append(("-" * len(title)).center(plot_width * 2))
    x_interval = x_data.max() - x_data.min()
    y_interval = y_data.max() - y_data.min()
    normalised_x_data = (((x_data - x_data.min()) * (plot_width - 1)) / x_interval).astype(int)
    normalised_y_data = (((y_data - y_data.min()) * (plot_height - 1)) / y_interval).astype(int)
    grid = np.full((plot_height, plot_width), fill_value=" ", dtype=object)
    grid[normalised_y_data, normalised_x_data] = "x"
    y_tick_padding = len(f"{y_data.max():.2f}")
    x_tick_gap = len(f"{x_data.max():.2f}") + 1
    string_builder.append("\n")
    for index, row in zip(range(plot_height, 0, -1), reversed(grid)):
        string_builder.append(f"{y_interval * (index / (plot_height - 1)) + y_data.min():<{y_tick_padding}.2f} | ")
        string_builder.extend(row, sep=" ")
        string_builder.append("\n")
    string_builder.append(" " * (y_tick_padding + 1))
    string_builder.append("=" * ((plot_width * 2) + 2))
    string_builder.append("\n")
    string_builder.append(" " * (y_tick_padding + 3))
    string_builder.extend("^" * math.floor((plot_width * 2) / (x_tick_gap + 1)), sep=(" " * x_tick_gap))
    string_builder.append("\n")
    string_builder.append(" " * (y_tick_padding + 3))
    string_builder.extend(f"{x_interval * (index / (plot_width - 1)) + x_data.min():<{x_tick_gap + 1}.2f}" for index in range(0, plot_width * 2, x_tick_gap + 1))
    return string_builder.compile()

print(make_terminal_relplot(np.arange(0, 200), np.linspace(1, 40, 200) ** 2, 100, 100, "Hello World"))
