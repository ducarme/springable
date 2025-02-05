import os
from pathlib import Path


def _adjust_spines(ax, offset, spines=("left", "bottom"), outward=True):
    for loc, spine in ax.spines.items():
        if loc in spines:
            ax.spines[loc].set_visible(True)
            if outward:
                spine.set_position(("outward", offset))
        else:
            ax.spines[loc].set_visible(False)  # don't draw spine
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    elif "right" in spines:
        ax.yaxis.set_ticks_position("right")
    else:
        # no yaxis ticks
        ax.yaxis.set_visible(False)

    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    elif "top" in spines:
        ax.xaxis.set_ticks_position("top")
    else:
        # no xaxis ticks
        ax.xaxis.set_visible(False)


def save_fig(fig, save_dir, save_name, formats, transparent=False, dpi=None):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if not isinstance(formats, list):
        formats = [formats]
    for format in formats:
        if dpi is None:
            fig.savefig(os.path.join(save_dir, save_name + '.' + format), transparent=transparent)
        else:
            fig.savefig(os.path.join(save_dir, save_name + '.' + format), transparent=transparent, dpi=dpi)



def adjust_spines(axs, offset):
    only_one_axis = not isinstance(axs, list)
    if only_one_axis:
        axs = [axs]
    for ax in axs:
        _adjust_spines(ax, offset)


def adjust_figure_layout(fig, fig_width=None, fig_height=None, pad=0.0):
    if fig_width is not None:
        fig.set_figwidth(fig_width)
    if fig_height is not None:
        fig.set_figheight(fig_height)
    fig.tight_layout(pad=pad)
