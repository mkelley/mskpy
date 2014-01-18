# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
graphics --- Helper functions for making plots.
===============================================

   arrows
   axcolor
   circle
   ds9
   harrows
   jdaxis2date
   ksplot
   nicelegend
   niceplot
   noborder
   remaxes
   tplot
   tplot_setup

"""

__all__ = [
   'arrows',
   'axcolor',
   'circle',
   'ds9',
   'harrows',
   'jdaxis2date',
   'ksplot',
   'nicelegend',
   'niceplot',
   'noborder',
   'remaxes',
   'tplot',
   'tplot_setup'
]

import numpy as np
import matplotlib.pyplot as plt

def arrows(xy, length, rot=0, angles=[0, 90], labels=['N', 'E'],
           offset=1.3, inset=0, fontsize='medium', arrowprops=dict(),
           **kwargs):
    """Draw arrows, E of N.

    Parameters
    ----------
    xy : array
      `(x, y)` location in data units for the base of the arrows.
    length : float
      Length of the arrows in data units.
    rot : float, optional
      The image orientation (position angle of north).
    angles : array, floats, optional
      The position angles at which to place arrows, measured E of N.
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

    ax = kwargs.pop('axes', plt.gca())

    if type(arrowprops) == dict:
        arrowprops['arrowstyle'] = arrowprops.pop('arrowstyle', '<-')
        arrowprops['shrinkB'] = arrowprops.pop('shrinkB', 0)

    alist = []
    for i in range(len(angles)):
        ixy = length * inset * np.array(
            -np.sin(rot + np.radians(angles[i])),
             np.cos(rot + np.radians(angles[i])))
        dxy = length * offset * np.array(
            -np.sin(rot + np.radians(angles[i])),
             np.cos(rot + np.radians(angles[i])))
        alist += [ax.annotate(labels[i], xy + ixy, xy + dxy,
                              ha='center', va='center',
                              fontsize=fontsize, arrowprops=arrowprops,
                              **kwargs)]
    return alist

def axcolor(color):
    """Sets the color of all future axis lines and labels.

    Parameters
    ----------
    color : string or tuple
      Any acceptable matplotlib color.

    """
    plt.rc('xtick', color=color)
    plt.rc('ytick', color=color)
    plt.rc('axes', edgecolor=color)
    plt.rc('axes', labelcolor=color)

def circle(x, y, r, segments=100, **kwargs):
    """Draw a circle.

    Parameters
    ----------
    x, y, r : float or array
      x coordinate, y coordinate, radius.
    segments : int, optional
      The number of line segements in the circle.
    **kwargs
      `matplotlib.plot` keywords.

    """

    if np.iterable(x) and np.iterable(y) and np.iterable(r):
        for i in xrange(len(x)):
            circle(x[i], y[i], r[i], **keywords)
    elif np.iterable(x) and np.iterable(y):
        for i in xrange(len(x)):
            circle(x[i], y[i], r, **keywords)
    elif np.iterable(r):
        for i in xrange(len(r)):
            circle(x, y, r[i], **keywords)
    else:
        th = np.linspace(0, 2 * np.pi, segments)
        xx = r * np.sin(th) + x
        yy = r * np.cos(th) + y
        plt.plot(xx, yy, **keywords)

def ds9(**kwargs):
    """Return a DS9 instance with a `view` method.

    `view` is a copy of `set_np2arr` for convenience.

    """
    try:
        import ds9
    except ImportError:
        print "Requires pyds9."
        raise
    disp = ds9.ds9(**kwargs)
    disp.view = disp.set_np2arr
    return disp

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
      Any valid `plt.annotate` or `plt.arrows` keyword argument
      (except rot).

    Returns
    -------
    alist : list
      List of items returned from `annotate`.

    """

    from .util import getrot

    rot = np.radians(getrot(header)[1])
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
    from util import jd2time
    jd = axis.get_ticklocs()
    return axis.set_ticklabels(
        [jd2time(t).datetime.strftime(fmt) for t in jd])

def ksplot(x, ax=None, **kwargs):
    """Graphical version of the Kolmogorov-Smirnov test.

    Parameters
    ----------
    x : array
      The dataset to plot.
    ax : matplotlib.axes
      Plot to this axis.
    **kwargs
      `matplotlib.plot` keywords for the dataset.  The default
      linestyle is "steps-post".

    Returns
    -------
    line : matplotlib.lines
      Output from `matplotlib.plot`.

    """

    xx = np.sort(x)
    yy = np.ones(x.size).cumsum() / float(x.size)
    ls = keywords.pop('ls', keywords.pop('linestyle', 'steps-post'))
    if ax is None:
        ax = plt.gca()
    return plt.plot(np.r_[xx[0], xx], np.r_[0, yy], ls=ls, **keywords)

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

    """

    from matplotlib.font_manager import FontProperties

    axis = kwargs.pop('axis', None)
    numpoints = kwargs.pop('numpoints', 1)
    fontsize = kwargs.pop('fontsize', 'medium')

    if axis is not None:
        plt.sca(axis)

    kwargs['prop'] = FontProperties(size=fontsize)
    kwargs['numpoints'] = numpoints

    return plt.legend(*args, **kwargs)


def niceplot(ax=None, axfs='large', lfs='x-large', tightlayout=True,
             **kwargs):
    """Pretty up a plot for publication.

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
            niceplot(ax, tightlayout=tightlayout, axfs=axfs, lfs=lfs, **kwargs)

    # for the axes
    plt.setp(ax.get_ymajorticklabels(), fontsize=axfs)
    plt.setp(ax.get_xmajorticklabels(), fontsize=axfs)

    # axis labes
    labels = (ax.xaxis.get_label(), ax.yaxis.get_label())
    plt.setp(labels, fontsize=lfs)

    # for plot markers, ticks
    mew = kwargs.pop('markeredgewidth', kwargs.pop('mew', 1.25))
    ms = kwargs.pop('markersize', kwargs.pop('ms', 7.0))
    lw = kwargs.pop('linewidth', kwargs.pop('lw', 2.0))

    plt.setp(ax.get_lines(), mew=mew, ms=ms, lw=lw, **kwargs)

    lines = ax.xaxis.get_minorticklines() + ax.xaxis.get_majorticklines() + \
        ax.yaxis.get_minorticklines() + ax.yaxis.get_majorticklines()
    plt.setp(lines, mew=1.25)

    # the frame
    plt.setp(ax.patch, lw=2.0)

    if hasattr(plt, "tight_layout") and tightlayout:
        plt.sca(ax)
        plt.tight_layout()

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
    linestyle = plotkws.pop('linestyle', plotkws.pop('ls', 'none'))

    x = lambda b, c: np.array(b) + np.array(c) / 2.0
    y = lambda c: np.array(c) * 0.86603

    if (erra is not None) or (errb is not None) or (errc is not None):
        points = []
        if erra is not None:
            # give each of the b- and c-axis 1/2 of the a uncertainty
            lb = b + 0.5 * erra[0] * b / (b + c)
            ub = b - 0.5 * erra[1] * b / (b + c)
            uc = c - 0.5 * erra[1] * c / (b + c)
            lc = c + 0.5 * erra[0] * c / (b + c)
            points.append(plt.plot(np.c_[x(lb, lc), x(ub, uc)].T,
                                   np.c_[y(lc), y(uc)].T,
                                   '-', color='0.5'))
        if errb is not None:
            lb = b - errb[0]
            ub = b + errb[1]
            # give the c-axis 1/2 of the b uncertainty (the other half
            # would be on the a-axis)
            lc = c + 0.5 * errb[0]
            uc = c - 0.5 * errb[1]
            points.append(plt.plot(np.c_[x(lb, lc), x(ub, uc)].T,
                                   np.c_[y(lc), y(uc)].T,
                                   '-', color='0.5'))
        if errc is not None:
            lc = c - errc[0]
            uc = c + errc[1]
            # give the b-axis 1/2 of the c uncertainty (the other half
            # would be on the a-axis)
            ub = b - 0.5 * errc[1]
            lb = b + 0.5 * errc[0]
            points.append(plt.plot(np.c_[x(lb, lc), x(ub, uc)].T,
                                   np.c_[y(lc), y(uc)].T,
                                   '-', color='0.5'))
        
        points.append(plt.plot(x(b, c), y(c), linestyle=linestyle, **plotkws))
    else:
        points = plt.plot(x(b, c), y(c), linestyle=linestyle, **plotkws)

    return points        

def tplot_setup(alabel=None, blabel=None, clabel=None,
                axes=dict(color='k', linestyle='-'),
                grid=dict(color='0.5', linestyle='--')):
    """Set up a ternary plot.

    a is the lower-left corner, b is the lower-right corner, c is the
    top.

    Parameters
    ----------
    alabel, blabel, clabel : str, optional
      Labels for these bases.
    axes : dictionary, optional
      Plot keywords for the axis lines, or None for no axes.
    grid : dictionary, optional
      Plot keywords for the grid lines, or None for no grid lines.

    """

    x = lambda b, c: np.array(b) + np.array(c) / 2.0
    y = lambda c: np.array(c) * 0.86603

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

        b = [0.25, 0.5, 0.75, 0.75, 0.5, 0.25, 0, 0, 0]
        c = [0, 0, 0, 0.25, 0.5, 0.75, 0.75, 0.5, 0.25]
        l = ['75/25', '50', '25/75', '25/75', '50', '75/25',
             '25/75', '50', '75/25']
        dx = [0, 0, 0, 0.015, 0.015, 0.015, -0.015, -0.015, -0.015]
        dy = [-0.021, -0.021, -0.021, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        a = [0, 0, 0, -60, -60, -60, 60, 60, 60]
        for i in range(len(b)):
            plt.annotate(l[i], (x(b[i], c[i]) + dx[i], y(c[i]) + dy[i]),
                       ha='center', va='center', rotation=a[i])

    if alabel is not None:
        plt.annotate(alabel, (-0.05, -0.04), ha='center',
                     va='baseline')
    if blabel is not None:
        plt.annotate(blabel, (1.05, -0.04), ha='center',
                   va='baseline')
    if clabel is not None:
        plt.annotate(clabel, (0.5, 0.9), ha='center',
                   va='baseline')

    plt.gcf().subplots_adjust(top=1.0, left=0, bottom=0, right=1.0)
    ax = plt.gca()
    ax.axis('off')
    ax.axis('equal')
    plt.setp(plt.gca(), ylim=(-0.1, 1), xlim=(-0.01, 1.01))

# update module docstring
from .util import autodoc
autodoc(globals())
del autodoc
