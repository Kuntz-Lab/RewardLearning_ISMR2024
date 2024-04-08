import random
import numpy as np
import seaborn as sns
from matplotlib import colors


def random_colors(n_colors, palette='hls', xkcd_colors=[], alpha=None):
    """
    Generates list of random RGB colors. If alpha is provided then it will give RGBA colors.

    Args:
        n_colors (int): Number of random colors to return.
        palette: Seaborn palette to generate random colors from. Can be a 
                 string identifying a palette (e.g. 'hls', 'husl', 'Set2'),
                 or a list of colors (e.g. ["#9b59b6", "#3498db", "#95a5a6"]).
        xkcd_colors (list): List of string color names from the xkcd RGB color
                            names. Overrides 'palette' arg.
        alpha (float): Alpha value in range [0,1], will force RGBA returns instead of RGB.
    Returns:
        colors (list): List of n_colors 3-tuples of RGB values.

    Options for xkcd_colors: https://xkcd.com/color/rgb
    
    Choosing Seaborn palettes: https://seaborn.pydata.org/tutorial/color_palettes.html
    """
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("Alpha value must be in range [0,1]")
    if xkcd_colors:
        palette = list(sns.xkcd_palette(xkcd_colors))
    colors = list(sns.color_palette(palette, n_colors))
    random.shuffle(colors)
    colors = [list(c) for c in colors]  # Lists are better than tuples, can append alphas
    if alpha is not None:
        colors = [c + [alpha] for c in colors]
    return colors


def get_color(color_name, alpha=None):
    """
    Returns a tuple (R,G,B) of float values in range [0,1]. If alpha 
    is provided then it will return (R,G,B,A).

    Color can be any name from this page:
    https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    converter = colors.ColorConverter()
    color = converter.to_rgb(colors.cnames[color_name])
    if alpha is not None:
        color = (*color, alpha)
    return list(color)


if __name__ == '__main__':
    print(get_color('bisque', 0.4))
