# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""finding --- Finding charts
=============================

Finding charts may be generated from the command line with `python3 -m
mskpy.observing.finding`.

   Functions
   ---------
   finding_chart

"""

import numpy as np

__all__ = [
    'finding_charts',
]


def finding_charts(target, observer, start,
                   surveys=['SDSSr', 'DSS Red', 'DSS'],
                   text=None, size=15, step=1, lstep=6, number=25,
                   lformat='%H:%M', alpha=0.5, lalpha=1, **kwargs):
    """Generate finding charts for a moving target.

    The downloaded images are saved as gzipped FITS with a name based
    on the ephemeris position.  If an image has already been
    downloaded for an ephemeris position, then it will read the file
    instead of fetching a new one.

    The finding charts are saved with the RA, Dec, and time of the
    center ephemeris point.

    Parameters
    ----------
    target : string
        Target designation.

    observer : string
        Observer location.

    date : array-like of string
        Start date for the finding charts.  Processed with
        `astropy.time.Time`.

    surveys : array-like, optional
        List of surveys to display in order of preference.  See
        `astroquery.skyview`.

    size : int, optional
        Field of view in arcmin.

    text : string, optiona
        Additional text to add to the upper-left corner of the plot.

    step : float, optional
        Length of each time step. [hr]

    lstep : float, optional
        Length of time steps between labels. [hr]

    number : int, optional
        Number of steps.

    lformat : string, optional
        Label format, using strftime format codes.

    alpha : float, optional
        Transparency for figure annotations.  0 to 1 for transparent
        to solid.

    lalpha : float, optional
        Transparency for labels.

    **kwargs
        Additional keyword arguments are passed to
        `astroquery.mpc.MPC.get_ephemeris`.

    """

    import os
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import skycoord_to_pixel
    from astropy.time import Time
    import astropy.units as u
    import astropy.coordinates as coords
    from astropy.table import vstack
    from astroquery.skyview import SkyView
    from astroquery.mpc import MPC
    import aplpy
    from .. import util

    # nominal tick marks
    t0 = Time(start)
    current = 0
    tab = None
    while True:
        t1 = t0 + current * u.hr
        n = min(number - current, 1440)
        if n <= 0:
            break

        e = MPC.get_ephemeris(target, location=observer, start=t1,
                              step=step * u.hr, number=n, **kwargs)
        if tab is None:
            tab = e
        else:
            tab = vstack((tab, e))

        current += n

    dates = tab['Date']
    eph = coords.SkyCoord(u.Quantity(tab['RA']), u.Quantity(tab['Dec']))

    # label tick marks
    current = 0
    tab = None
    number_lsteps = int(number * step / lstep)
    while True:
        t1 = t0 + current * u.hr
        n = min(number_lsteps - current, 1440)
        if n <= 0:
            break

        e = MPC.get_ephemeris(target, location=observer, start=t1,
                              step=lstep * u.hr, number=n, **kwargs)
        if tab is None:
            tab = e
        else:
            tab = vstack((tab, e))

        current += n

    dates_labels = tab['Date']
    labels = coords.SkyCoord(u.Quantity(tab['RA']), u.Quantity(tab['Dec']))

    step = 0
    while step < len(eph):
        print('This step: ', dates[step],
              eph[step].to_string('hmsdms', sep=':', precision=0))

        fn = '{}'.format(
            eph[step].to_string('hmsdms', sep='', precision=0).replace(' ', ''))

        if os.path.exists('sky-{}.fits.gz'.format(fn)):
            print('reading from file')
            im = fits.open('sky-{}.fits.gz'.format(fn))
        else:
            print('reading from URL')
            # note, radius seems to be size of image
            position = eph[step].to_string('decimal', precision=6)
            opts = dict(position=position, radius=size * u.arcmin,
                        pixels=int(900 / 15 * size))

            im = None
            for survey in surveys:
                print(survey)
                try:
                    im = SkyView.get_images(survey=survey, **opts)[0]
                except:
                    continue
                break

            if im is None:
                raise ValueError("Cannot download sky image.")

            im.writeto('sky-{}.fits.gz'.format(fn))

        wcs = WCS(im[0].header)
        c = skycoord_to_pixel(eph, wcs, 0)
        i = (c[0] > 0) * (c[0] < 900) * (c[1] > 0) * (c[1] < 900)

        c = skycoord_to_pixel(labels, wcs, 0)
        j = (c[0] > 0) * (c[0] < 900) * (c[1] > 0) * (c[1] < 900)

        fig = aplpy.FITSFigure(im)
        fig.set_title(target)
        vmin = np.min(im[0].data)
        if vmin < 0:
            vmid = vmin * 1.5
        else:
            vmid = vmin / 2
        fig.show_colorscale(cmap='viridis', stretch='log', vmid=vmid)
        fig.show_lines([np.vstack((eph[i].ra.degree, eph[i].dec.degree))],
                       color='w', alpha=alpha)
        fig.show_markers(eph[i].ra.degree, eph[i].dec.degree, c='w',
                         marker='x', alpha=alpha)

        for k in np.flatnonzero(j):
            d = dates_labels[k]
            fig.add_label(labels.ra[k].degree, labels.dec[k].degree,
                          d.datetime.strftime(lformat), color='w',
                          alpha=lalpha, size='small')

        fig.show_rectangles(eph[step].ra.degree, eph[step].dec.degree,
                            1 / 60, 1 / 60, edgecolors='w', alpha=alpha)

        if text is not None:
            ax = plt.gca()
            ax.text(0.03, 0.98, text, va='top', transform=ax.transAxes,
                    size=16, color='w')

        t = target.replace(' ', '').replace('/', '').replace("'", '').lower()
        d = dates[step]
        d = d.isot[:16].replace('-', '').replace(':', '').replace('T', '_')
        fig.save('{}-{}-{}.png'.format(t, d, fn),
                 dpi=300)

        step = np.flatnonzero(i)[-1] + 1


if __name__ == "__main__":
    import argparse
    from astropy.time import Time
    from .. import ephem

    parser = argparse.ArgumentParser(
        description='Moving target finding charts.')
    parser.add_argument('target', nargs='+', help='Name of a target.')
    parser.add_argument('start', help='Start date.')
    parser.add_argument('end', help='End date.')
    parser.add_argument('--observer', default='500', help='Observer location.')
    parser.add_argument('--step', default=1, type=float,
                        help='Tick mark step size in hours.')
    parser.add_argument('--lstep', default=6, type=float,
                        help='Label step size in hours.')
    parser.add_argument('--lformat', default='%H:%M',
                        help='Label format, using strftime codes.')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Amount of transparency for overlays.')
    parser.add_argument('--surveys', default='SDSSr,DSS2 Red,DSS1 Red,DSS',
                        help='Comma-separated list of SkyView surveys to query.')
    parser.add_argument('--size', default=15, type=int,
                        help='Requested FOV size in arcmin.')

    args = parser.parse_args()
    target = ' '.join(args.target)
    assert len(target.strip()) > 0
    number = int((Time(args.end).jd - Time(args.start).jd) * 24 / args.step)
    finding_charts(target, args.observer, args.start, size=args.size,
                   surveys=args.surveys.split(','), number=number,
                   step=args.step, lstep=args.lstep,
                   lformat=args.lformat, alpha=args.alpha)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
