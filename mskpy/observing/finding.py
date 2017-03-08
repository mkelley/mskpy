# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""finding --- Finding charts
==========================

Finding charts may be generated from the command line with `python3 -m
mskpy.observing.finding`.

   Functions
   ---------
   finding_chart

"""

import numpy as np

__all__ = [
    'finding_chart',
]

def finding_charts(target, observer, dates, step=1, lstep=6,
                  alpha=0.5):
    """Generate finding charts for a moving target.

    The downloaded images are saved as gzipped FITS with a name based
    on the ephemeris position.  If an image has already been
    downloaded for an ephemeris position, then it will read the file
    instead of fetching a new one.

    The finding charts are saved with the RA, Dec, and time of the
    center ephemeris point.

    Parameters
    ----------
    target : SolarSysObject
      The target.
    observer : SolarSysObject
      The observer.
    date : array-like of string
      The start and end dates for the finding charts.  Processed with
      `util.date2time`.
    step : float, optional
      Length of each time step. [hr]
    lstep : float, optional
      Length of time steps between labels. [hr]
    alpha : float, optional
      Transparency for figure annotations.  0 to 1 for transparent to
      solid.

    """

    import os
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u
    import astropy.coordinates as coords
    from astroquery.skyview import SkyView
    import aplpy
    from .. import util

    # dates
    start, stop = util.date2time(dates)
    jd = np.arange(start.jd, stop.jd + step / 24, step / 24)
    jd_labels = np.arange(start.jd, stop.jd + lstep / 24, lstep / 24)

    # geometry
    g = observer.observe(target, jd, ltt=True)
    g_labels = observer.observe(target, jd_labels, ltt=True)

    eph = coords.SkyCoord(g.ra, g.dec)
    
    step = 0
    while step < len(g):
        print('This step: ', util.date2time(jd[step]).iso,
              eph[step].to_string('hmsdms', sep=':', precision=0))

        fn = '{}'.format(
            eph[step].to_string('hmsdms', sep='', precision=0).replace(' ', ''))

        if os.path.exists('sky-{}.fits.gz'.format(fn)):
            print('reading from file')
            im = fits.open('sky-{}.fits.gz'.format(fn))
        else:
            print('reading from URL')
            # note, radius seems to be size of image
            opts = dict(position=eph[step], radius=15 * u.arcmin, pixels=900)

            print('Trying SDSS.')
            try:
                im = SkyView.get_images(survey='SDSSr', **opts)[0]
            except:
                im = None
            
            if im is not None:
                if np.sum(im[0].data == 0) / np.prod(im[0].data.shape) > 0.2:
                    print('  SDSS image coverage is too low.')
                    im = None

            if im is None:
                # try DSS2
                im = SkyView.get_images(survey='DSS2 Red', **opts)[0]

            if im is None:
                # last resort: DSS1
                im = SkyView.get_images(survey='DSS1 Red', **opts)[0]

            assert im is not None, "Cannot download sky image, tried SDSSr, DSS2 Red, and DSS1 Red from NASA SkyView."

            im.writeto('sky-{}.fits.gz'.format(fn))

        wcs = WCS(im[0].header)
        c = wcs.wcs_world2pix(g.ra.degree, g.dec.degree, 0)
        i = (c[0] > 0) * (c[0] < 900) * (c[1] > 0) * (c[1] < 900)

        c = wcs.wcs_world2pix(g_labels.ra.degree, g_labels.dec.degree, 0)
        j = (c[0] > 0) * (c[0] < 900) * (c[1] > 0) * (c[1] < 900)

        fig = aplpy.FITSFigure(im)
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
            d = util.date2time(jd_labels[k])
            fig.add_label(g_labels.ra[k].degree, g_labels.dec[k].degree,
                          d.datetime.strftime('%H:%M'), color='w',
                          alpha=alpha, size='small')

        fig.show_rectangles(eph[step].ra.degree, eph[step].dec.degree,
                            1 / 60, 1 / 60, edgecolors='w', alpha=alpha)
        t = target.name.replace(' ', '').replace('/', '').replace("'", '').lower()
        d = util.date2time(jd[step])
        fig.save('{}-{}-{}.png'.format(t, fn, d.isot[:16].replace(':', '')),
                 dpi=300)

        step = np.flatnonzero(i)[-1] + 1

if __name__ == "__main__":
    import argparse
    from .. import ephem

    parser = argparse.ArgumentParser(description='Moving target finding charts.')
    parser.add_argument('target', nargs='+', help='Name of a target.')
    parser.add_argument('start', help='Start date.')
    parser.add_argument('end', help='End date.')
    parser.add_argument('--observer', default='Earth', help='Name of observer.')
    parser.add_argument('--step', default=1, type=float, help='Tick mark step size in hours.')
    parser.add_argument('--lstep', default=6, type=float, help='Label step size in hours.')
    parser.add_argument('--alpha', default=0.5, type=float, help='Amount of transparency for overlays.')

    args = parser.parse_args()

    if (len(args.target) == 1
        and args.target[0].isdigit()
        and len(args.target[0]) < 6):
        # asteroid designation
        target = ephem.getspiceobj(int(args.target[0]) + 2000000)
    else:
        target = ephem.getspiceobj(' '.join(args.target))

    try:
        observer = eval('ephem.' + args.observer.capitalize())
    except AttributeError:
        observer = ephem.getspiceobj(' '.join(args.observer))

    finding_charts(target, observer, [args.start, args.end],
                   step=args.step, lstep=args.lstep, alpha=args.alpha)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
