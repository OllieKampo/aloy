
import math
import warnings
import numpy as np

from auxiliary.stringutils import StringBuilder

def make_terminal_relplot(x_data: np.ndarray,
                          y_data: np.ndarray,
                          plot_width: int,
                          plot_height: int,
                          title: str,
                          markers: list[str] = ["x", "o", "+"],
                          legend: list[str] | None = None
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
        warnings.warn("x_data has fewer elements than the plot width. Plot will be sparse.")
    if y_data.shape[-1] < plot_height:
        warnings.warn("y_data has fewer elements than the plot height. Plot will be sparse.")
    
    ## Create the string builder and add the title
    string_builder = StringBuilder(title.center(plot_width * 2))
    string_builder.append("\n")
    string_builder.append(("-" * (len(title) + 2)).center(plot_width * 2))
    if legend is not None:
        if len(legend) != x_data.shape[0]:
            raise ValueError("Legend must have the same number of elements as x_data has rows")
        string_builder.append("\n")
        string_builder.append((" || ".join((f"{label} [{markers[i % len(markers)]}]"
                                            for i, label in enumerate(legend)))).center(plot_width * 2))
    
    ## Normalise the data to fit the plot
    x_interval = x_data.max() - x_data.min()
    y_interval = y_data.max() - y_data.min()
    normalised_x_data = (((x_data - x_data.min()) * (plot_width - 1)) / x_interval).astype(int)
    normalised_y_data = (((y_data - y_data.min()) * (plot_height - 1)) / y_interval).astype(int)

    ## Create the plot grid, and fill it with the data points
    grid = np.full((plot_height, plot_width), fill_value=" ", dtype=object)
    if normalised_x_data.ndim == 1:
        grid[normalised_y_data, normalised_x_data] = "x"
    else:
        for i in range(normalised_x_data.shape[0]):
            grid[normalised_y_data[i], normalised_x_data[i]] = markers[i % len(markers)]
    
    ## Add the y-axis, ticks and labels, and the plot itself
    y_tick_padding = len(f"{y_data.max():.2f}")
    x_tick_gap = len(f"{x_data.max():.2f}") + 1
    string_builder.append("\n")
    for index, row in zip(range(plot_height, 0, -1), reversed(grid)):
        string_builder.append(f"{y_interval * (index / (plot_height - 1)) + y_data.min():<{y_tick_padding}.2f} | ")
        string_builder.extend(row, sep=" ")
        string_builder.append("\n")
    
    ## Add the x-axis, ticks and labels
    string_builder.append(" " * (y_tick_padding + 1))
    string_builder.append("=" * ((plot_width * 2) + 2))
    string_builder.append("\n")
    string_builder.append(" " * (y_tick_padding + 3))
    string_builder.extend("^" * math.floor((plot_width * 2) / (x_tick_gap + 1)), sep=(" " * x_tick_gap))
    string_builder.append("\n")
    string_builder.append(" " * (y_tick_padding + 3))
    string_builder.extend(f"{(x_interval * (index / (plot_width - 1))) + x_data.min():<{x_tick_gap + 1}.2f}" for index in range(0, plot_width, (x_tick_gap + 1) // 2))

    return string_builder.compile()

if __name__ == "__main__":
    print(make_terminal_relplot(np.arange(0, 200), np.linspace(1, 40, 200) ** 1.5, 40, 20, "Hello World"))
    print(make_terminal_relplot(np.tile(np.arange(0, 200), 3).reshape(3, 200),
                                np.array([np.linspace(0, 40, 200), (np.linspace(0, 40, 200) ** 1.5), (np.linspace(0, 40, 200) ** 2)]),
                                40, 20, "Hello World", legend=["y = x * 5", "y = (x * 5) ^ 1.5", "y = (x * 5) ^ 2"]))
