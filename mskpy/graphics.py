# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
graphics --- Helper functions for making plots.
===============================================

   arrows
   axcolor
   circle
   harrows
   jdaxis2date
   ksplot
   nicelegend
   niceplot
   noborder
   remaxes
   rem_interior_ticklabels
   savepdf2pdf
   tplot
   tplot_setup

"""

from .util import autodoc

__all__ = [
    "arrows",
    "axcolor",
    "circle",
    "harrows",
    "jdaxis2date",
    "ksplot",
    "nicelegend",
    "niceplot",
    "noborder",
    "remaxes",
    "rem_interior_ticklabels",
    "savepdf2pdf",
    "tplot",
    "tplot_setup",
]

import os
from tempfile import NamedTemporaryFile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def arrows(
    xy,
    length,
    rot=0,
    angles=[0, 90],
    labels=["N", "E"],
    offset=1.3,
    inset=0,
    fontsize="medium",
    arrowprops=dict(),
    **kwargs
):
    """Draw arrows, E of N.

    Parameters
    ----------
    xy : array
      `(x, y)` location in data units for the base of the arrows.
    length : float
      Length of the arrows in data units.
    rot : float, optional
      The image orientation (position angle of north) in units of degrees.
    angles : array, floats, optional
      The position angles at which to place arrows, measured E of N,
      in units of degrees.
    labels : array, strings, optional
      Labels for each arrow, or None for no labels.
    offset : float, optional
      A length scale factor used to determine the placement of the
      labels.
    inset : float, optional
      A length scale factor used to determine the start position of
      the line along the vector.
    fontsize : string or float, optional
      Text fontsize (see `matplotlib.pyplot.annotate`).
    arrowprops : dict, optional
      Arrow properties (see `matplotlib.pyplot.annotate`).
    **kwargs
      Any valid `annotate` keyword argument.

    Returns
    -------
    alist : list
      List of items returned from `annotate`.

    """

    ax = kwargs.pop("axes", plt.gca())

    if type(arrowprops) == dict:
        arrowprops["arrowstyle"] = arrowprops.pop("arrowstyle", "<-")
        arrowprops["shrinkB"] = arrowprops.pop("shrinkB", 0)

    if labels is None:
        labels = [""] * len(angles)

    alist = []
    for i in range(len(angles)):
        a = np.radians(rot + angles[i])
        ixy = length * inset * np.array([-np.sin(a), np.cos(a)])
        dxy = length * offset * np.array([-np.sin(a), np.cos(a)])
        alist += [
            ax.annotate(
                labels[i],
                xy + ixy,
                xy + dxy,
                ha="center",
                va="center",
                fontsize=fontsize,
                arrowprops=arrowprops,
                **kwargs
            )
        ]
    return alist


def axcolor(color):
    """Sets the color of all future axis lines and labels.

    Parameters
    ----------
    color : string or tuple
      Any acceptable matplotlib color.

    """

    plt.rc("xtick", color=color)
    plt.rc("ytick", color=color)
    plt.rc("axes", edgecolor=color)
    plt.rc("axes", labelcolor=color)


def circle(x, y, r, ax=None, segments=100, **kwargs):
    """Draw a circle.

    Parameters
    ----------
    x, y, r : float or array
      x coordinate, y coordinate, radius.
    ax : matplotlib Axes
      Plot on this axis.
    segments : int, optional
      The number of line segements in the circle.
    **kwargs
      `matplotlib.plot` keywords.

    """

    if ax is None:
        ax = plt.gca()

    if np.iterable(x) and np.iterable(y) and np.iterable(r):
        for i in range(len(x)):
            circle(x[i], y[i], r[i], ax=ax, **kwargs)
    elif np.iterable(x) and np.iterable(y):
        for i in range(len(x)):
            circle(x[i], y[i], r, ax=ax, **kwargs)
    elif np.iterable(r):
        for i in range(len(r)):
            circle(x, y, r[i], ax=ax, **kwargs)
    else:
        th = np.linspace(0, 2 * np.pi, segments)
        xx = r * np.sin(th) + x
        yy = r * np.cos(th) + y
        ax.plot(xx, yy, **kwargs)


def harrows(header, xy, length, **kwargs):
    """Draw arrows based on the given FITS header.

    Parameters
    ----------
    header : astropy.fits.Header or string
      A FITS header object or name of a file.
    xy : array
      `(x, y)` location in data units for the base of the arrows.
    length : float or 2-element array
      Length of the arrows in data units.
    **kwargs
      Any valid `plt.annotate` or `mskpy.arrows` keyword argument
      (except rot).

    Returns
    -------
    alist : list
      List of items returned from `annotate`.

    """

    from .util import getrot

    rot = getrot(header)[1]
    return arrows(xy, length, rot=rot, **kwargs)


def jdaxis2date(axis, fmt):
    """Format a Julian Date axis tick labels as calendar date.

    Parameters
    ----------
    axis : matplotlib axis
    fmt : string
      The format of the tick labels.  See
      ``datetime.datetime.strftime``.

    Returns
    -------
    labels : matplotlib tick labels

    """
    from .util import jd2time

    jd = axis.get_ticklocs()
    return axis.set_ticklabels([jd2time(t).datetime.strftime(fmt) for t in jd])


def ksplot(x, xmax=None, ax=None, **kwargs):
    """Graphical version of the Kolmogorov-Smirnov test.

    Parameters
    ----------
    x : array
      The dataset to plot.
    xmax : float
      The maximal x value.  If provided, then a final line will be
      drawn from `x[-1]` to `xmax` along `y=1.0`.
    ax : matplotlib.axes
      Plot to this axis.
    **kwargs
      `matplotlib.plot` keywords for the dataset.

    Returns
    -------
    line : matplotlib.lines
      Output from `matplotlib.plot`.

    """

    xx = np.sort(x)
    yy = np.ones(xx.size).cumsum() / xx.size

    if xmax is None:
        xx = np.r_[xx[0], xx]
        yy = np.r_[0, yy]
    else:
        xx = np.r_[xx[0], xx, xmax]
        yy = np.r_[0, yy, 1]

    if ax is None:
        ax = plt.gca()

    return ax.step(xx, yy, where="post", **kwargs)


def nicelegend(*args, **kwargs):
    """A pretty legend for publications.

    Parameters
    ----------
    *args
      matplotlib.legend() arguments.
    **kwargs
      Any legend keyword argument.

    Returns
    -------
    leg : matplotlib.legend.Legend
      The drawn legend.

    Notes
    -----
    Remember that font properties are passed as a dictionary via the
    `prop` keyword.

    """

    axis = kwargs.pop("axis", None)

    kwargs["numpoints"] = kwargs.pop("numpoints", 1)

    prop = dict(size="medium")
    prop.update(kwargs.pop("prop", dict()))
    kwargs["prop"] = prop

    if axis is not None:
        plt.sca(axis)

    return plt.legend(*args, **kwargs)


def niceplot1(
    ax=None,
    axfs="12",
    lfs="14",
    tightlayout=True,
    mew=1.25,
    lw=2.0,
    ms=7.0,
    **kwargs
):
    """Clean up a plot for publication (matplotlib 1.x).

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot, optional
      An axis to niceify.  Default is all axes in the current figure.
    axfs : string, float, or int, optional
      Axis tick label font size.
    lfs : string, float, or int, optional
      Axis label font size.
    tightlayout : bool, optional
      Run `plt.tight_layout`.
    **kwargs
      Any line or marker property keyword.

    """

    if ax is None:
        for ax in plt.gcf().get_axes():
            niceplot(
                ax,
                tightlayout=tightlayout,
                axfs=axfs,
                lfs=lfs,
                mew=mew,
                lw=lw,
                ms=ms,
                **kwargs
            )

    # axis ticks
    plt.setp(ax.get_ymajorticklabels(), fontsize=axfs)
    plt.setp(ax.get_xmajorticklabels(), fontsize=axfs)

    # axis labels
    labels = (ax.xaxis.get_label(), ax.yaxis.get_label())
    plt.setp(labels, fontsize=lfs)

    # for plot markers, ticks
    lines = ax.get_lines()
    mew = kwargs.pop("markeredgewidth", kwargs.pop("mew", None))
    if mew is not None:
        plt.setp(lines, mew=mew)

    ms = kwargs.pop("markersize", kwargs.pop("ms", None))
    if ms is not None:
        plt.setp(lines, ms=ms)

    lw = kwargs.pop("linewidth", kwargs.pop("lw", None))
    if lw is not None:
        plt.setp(lines, lw=lw)

    if len(kwargs) > 0:
        plt.setp(lines, **kwargs)

    lines = (
        ax.xaxis.get_minorticklines()
        + ax.xaxis.get_majorticklines()
        + ax.yaxis.get_minorticklines()
        + ax.yaxis.get_majorticklines()
    )
    plt.setp(lines, mew=1.25)

    # the frame
    plt.setp(ax.patch, lw=2.0)

    if hasattr(plt, "tight_layout") and tightlayout:
        plt.sca(ax)
        plt.tight_layout()


def niceplot2(
    ax=None,
    tick_fs=12,
    label_fs=14,
    tight_layout=True,
    set_axis_formatter=True,
    **kwargs
):
    """Clean up a plot for publication.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot, optional
      An axis to niceify.  Default is all axes in the current figure.
    tick_fs : string, float, or int, optional
      Axis major tick label font size.  Minor ticks will be 1/sqrt(2)
      smaller.
    label_fs : string, float, or int, optional
      Axis label font size.
    tight_layout : bool, optional
      Run `plt.tight_layout`.
    **kwargs
      Any line or marker property keyword.

    """

    if ax is None:
        for ax in plt.gcf().get_axes():
            niceplot(
                ax,
                tight_layout=tight_layout,
                tick_fs=tick_fs,
                label_fs=label_fs,
                **kwargs
            )

    # axis ticks
    if isinstance(tick_fs, str):
        if tick_fs.isdecimal():
            minortick_fs = float(tick_fs) / np.sqrt(2)
        else:
            sizes = [
                "xx-small",
                "x-small",
                "small",
                "medium",
                "large",
                "x-large",
                "xx-large",
            ]
            assert tick_fs.lower() in sizes, "Unknown tick font size name"
            minortick_fs = sizes[max(0, sizes.index(tick_fs) - 1)]
    else:
        minortick_fs = tick_fs / np.sqrt(2)

    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.tick_params(axis="both", which="minor", labelsize=minortick_fs)

    # axis labels
    labels = (ax.xaxis.get_label(), ax.yaxis.get_label())
    plt.setp(labels, fontsize=label_fs)

    # for plot markers, ticks
    lines = ax.get_lines()
    #    mew = kwargs.pop('markeredgewidth', kwargs.pop('mew', None))
    #    if mew is not None:
    #        plt.setp(lines, mew=mew)
    #
    #    ms = kwargs.pop('markersize', kwargs.pop('ms', None))
    #    if ms is not None:
    #        plt.setp(lines, ms=ms)
    #
    #    lw = kwargs.pop('linewidth', kwargs.pop('lw', None))
    #    if lw is not None:
    #        plt.setp(lines, lw=lw)

    if len(kwargs) > 0:
        plt.setp(lines, **kwargs)

    # lines = ax.xaxis.get_minorticklines() + ax.xaxis.get_majorticklines() + \
    #    ax.yaxis.get_minorticklines() + ax.yaxis.get_majorticklines()
    # plt.setp(lines, mew=1.25)

    # the frame
    # plt.setp(ax.patch, lw=2.0)

    if tight_layout:
        plt.sca(ax)
        plt.tight_layout()


def niceplot(
    ax=None,
    tick_fs=8,
    label_fs=9,
    tight_layout=True,
    set_axis_formatter=True,
    **kwargs
):
    """Clean up a plot for publication.

    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot, optional
      An axis to niceify.  Default is all axes in the current figure.
    tick_fs : string, float, or int, optional
      Axis major tick label font size.  Minor ticks will be 1/sqrt(2)
      smaller.
    label_fs : string, float, or int, optional
      Axis label font size.
    tight_layout : bool, optional
      Run `plt.tight_layout`.
    **kwargs
      Any line or marker property keyword.

    """

    if ax is None:
        for ax in plt.gcf().get_axes():
            niceplot(
                ax,
                tight_layout=tight_layout,
                tick_fs=tick_fs,
                label_fs=label_fs,
                **kwargs
            )

    # axis ticks
    if isinstance(tick_fs, str):
        if tick_fs.isdecimal():
            minortick_fs = float(tick_fs) / np.sqrt(2)
        else:
            sizes = [
                "xx-small",
                "x-small",
                "small",
                "medium",
                "large",
                "x-large",
                "xx-large",
            ]
            assert tick_fs.lower() in sizes, "Unknown tick font size name"
            minortick_fs = sizes[max(0, sizes.index(tick_fs) - 1)]
    else:
        minortick_fs = tick_fs / np.sqrt(2)

    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.tick_params(axis="both", which="minor", labelsize=minortick_fs)

    # axis labels
    labels = (ax.xaxis.get_label(), ax.yaxis.get_label())
    plt.setp(labels, fontsize=label_fs)
    plt.tick_params("both", which="both", top=True, right=True)
    plt.tick_params("both", which="major", length=6)
    plt.tick_params("both", which="minor", length=3)

    # for plot markers, ticks
    lines = ax.get_lines()

    ax.minorticks_on()

    if len(kwargs) > 0:
        plt.setp(lines, **kwargs)

    if tight_layout:
        plt.sca(ax)
        plt.tight_layout(pad=0.5)


def noborder(fig=None):
    """Remove the figure border.

    Sets left = bottom = 0, right = top = 1, wspace = hspace = 0.

    Parameters
    ----------
    fig : matplotlib figure, optional
      Use this figure.

    """

    if fig is None:
        fig = plt.gcf()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)


def remaxes(ax=None):
    """Remove the axis lines.

    Parameters
    ----------
    ax : matplotlib axis, optional
      Use this axis.

    """

    if ax is None:
        ax = plt.gca()
    plt.setp(ax, frame_on=False, xticks=[], yticks=[])


def rem_interior_ticklabels(fig=None, axes=None, top=False, right=False):
    """Remove interior ticklabels from a multiaxis plot.

    Parameters
    ----------
    fig : matplotlib Figure, optional
      Inspect this figure for axes, otherwise use the current figure.
    axes : matplotlib axis, optional
      Only consider these axes, otherwise consider all axes in `fig`.
    top : bool, optional
      Set to `True` if the axes have ticks along the top.
    right : bool, optional
      Set to `True` if the axes have ticks along the right.

    """

    if fig is None:
        fig = plt.gcf()

    if axes is None:
        axes = fig.axes

    for ax in axes:
        if top:
            if not ax.is_first_row():
                ax.set_xticklabels([])
        else:
            if not ax.is_last_row():
                ax.set_xticklabels([])
        if right:
            if not ax.is_last_col():
                ax.set_yticklabels([])
        else:
            if not ax.is_first_col():
                ax.set_yticklabels([])


def savepdf2pdf(filename, **kwargs):
    """Save figure as pdf, then process with pdf2pdf.

    On my system, funny things happen with markers that have `alpha !=
    0` when vied with mupdf.  pdf2pdf (ghostscript) fixes it.

    Parameters
    ----------
    filename : string
      The name of the file to save.
    **kwargs
      Any `matplotlib.pyplot.savefig` keywords.

    """

    assert isinstance(filename, str)

    name = ""
    with NamedTemporaryFile(delete=False) as outf:
        name = outf.name
        plt.savefig(outf, format="pdf")

    os.system("pdf2pdf {} {}".format(name, filename))
    os.system("rm {}".format(name))


def tplot(b, c, erra=None, errb=None, errc=None, setup=False, **kwargs):
    """Plot data on a ternary plot.

    a is the lower-left corner, b is the lower-right corner, c is the
    top.

    Parameters
    ----------
    b, c : array
      The values to plot.  Along with `a`, these correspond to the 3
      bases of the ternary plot.  `a + b + c` must equal 1.0.
    erra, errb, errc : array, optional
      `2 x n` element arrays of the lower and upper error bars.
    setup : bool, optional
      Set to `True` to call `tplot_setup`.
    **kwargs
      Any `matplotlib.pyplot.plot` keywords.

    Returns
    -------
    points : list
      The return value from `plot`.

    Examples
    --------
    from np.random import rand
    from matplotlib.pyplot import clf, show
    from mskpy.graphics import tplot

    a = rand(100)
    b = rand(100) * (1.0 - a)
    c = 1.0 - a - b

    clf()
    tplot(b, c, setup=True)
    show()

    """

    if setup:
        tplot_setup()
    linestyle = kwargs.pop("linestyle", kwargs.pop("ls", "none"))

    def x(b, c):
        return np.array(b) + np.array(c) / 2.0

    def y(c):
        return np.array(c) * 0.86603

    if (erra is not None) or (errb is not None) or (errc is not None):
        points = []
        if erra is not None:
            # give each of the b- and c-axis 1/2 of the a uncertainty
            lb = b + 0.5 * erra[0] * b / (b + c)
            ub = b - 0.5 * erra[1] * b / (b + c)
            uc = c - 0.5 * erra[1] * c / (b + c)
            lc = c + 0.5 * erra[0] * c / (b + c)
            points.append(
                plt.plot(
                    np.c_[x(lb, lc), x(ub, uc)].T,
                    np.c_[y(lc), y(uc)].T,
                    "-",
                    color="0.5",
                )
            )
        if errb is not None:
            lb = b - errb[0]
            ub = b + errb[1]
            # give the c-axis 1/2 of the b uncertainty (the other half
            # would be on the a-axis)
            lc = c + 0.5 * errb[0]
            uc = c - 0.5 * errb[1]
            points.append(
                plt.plot(
                    np.c_[x(lb, lc), x(ub, uc)].T,
                    np.c_[y(lc), y(uc)].T,
                    "-",
                    color="0.5",
                )
            )
        if errc is not None:
            lc = c - errc[0]
            uc = c + errc[1]
            # give the b-axis 1/2 of the c uncertainty (the other half
            # would be on the a-axis)
            ub = b - 0.5 * errc[1]
            lb = b + 0.5 * errc[0]
            points.append(
                plt.plot(
                    np.c_[x(lb, lc), x(ub, uc)].T,
                    np.c_[y(lc), y(uc)].T,
                    "-",
                    color="0.5",
                )
            )

        points.append(plt.plot(x(b, c), y(c), linestyle=linestyle, **kwargs))
    else:
        points = plt.plot(x(b, c), y(c), linestyle=linestyle, **kwargs)

    return points


def tplot_setup(
    alabel=None,
    blabel=None,
    clabel=None,
    axes=dict(color="k", linestyle="-"),
    grid=dict(color="0.5", linestyle="--"),
):
    """Set up a ternary plot.

    a is the lower-left corner & base, b is the lower-right
    corner & right side, c is the top corner & left side.

    Parameters
    ----------
    alabel, blabel, clabel : str, optional
      Labels for these bases.
    axes : dictionary, optional
      Plot keywords for the axis lines, or None for no axes.
    grid : dictionary, optional
      Plot keywords for the grid lines, or None for no grid lines.

    Returns
    -------
    ax : matplotlib axes

    """

    def x(b, c):
        return np.array(b) + np.array(c) / 2.0

    def y(c):
        return np.array(c) * 0.86603

    if axes is not None:
        b = [0, 1, 0, 0]
        c = [0, 0, 1, 0]
        plt.plot(x(b, c), y(c), **axes)

    if grid is not None:
        b = [0, 0.25, 0.25, 0, 0.75, 0.75, 0]
        c = [0.25, 0, 0.75, 0.75, 0, 0.25, 0.25]
        plt.plot(x(b, c), y(c), **grid)

        b = [0, 0.5, 0.5, 0]
        c = [0.5, 0, 0.5, 0.5]
        plt.plot(x(b, c), y(c), **grid)

        b = [0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25, 0, 0, 0, 0, 0, 0]
        c = [0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25, 0]
        l = [
            "100",
            "75",
            "50",
            "25",
            "0",  # a
            "100",
            "75",
            "50",
            "25",
            "0",  # b
            "100",
            "75",
            "50",
            "25",
            "0",
        ]  # c
        dx = [
            0,
            0,
            0,
            0,
            0,
            0.015,
            0.015,
            0.015,
            0.015,
            0.015,
            -0.015,
            -0.015,
            -0.015,
            -0.015,
            -0.015,
        ]
        dy = [
            -0.021,
            -0.021,
            -0.021,
            -0.021,
            -0.021,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ]
        a = [0, 0, 0, 0, 0, -60, -60, -60, -60, -60, 60, 60, 60, 60, 60]
        for i in range(len(b)):
            plt.annotate(
                l[i],
                (x(b[i], c[i]) + dx[i], y(c[i]) + dy[i]),
                ha="center",
                va="center",
                rotation=a[i],
            )

    if alabel is not None:
        plt.text(
            x(0.5, 0),
            y(0) - 0.05,
            alabel,
            ha="center",
            va="center",
            fontsize=14,
        )
    if blabel is not None:
        plt.text(
            x(0.5, 0.5) + 0.04,
            y(0.5) + 0.03,
            blabel,
            ha="center",
            va="center",
            rotation=-60,
            fontsize=14,
        )
    if clabel is not None:
        plt.text(
            x(0, 0.5) - 0.04,
            y(0.5) + 0.03,
            clabel,
            ha="center",
            va="center",
            rotation=60,
            fontsize=14,
        )

    plt.gcf().subplots_adjust(top=1.0, left=0, bottom=0, right=1.0)
    ax = plt.gca()
    ax.axis("off")
    ax.axis("equal")
    plt.setp(plt.gca(), ylim=(-0.1, 1), xlim=(-0.01, 1.01))
    return ax


# update module docstring
autodoc(globals())
del autodoc
