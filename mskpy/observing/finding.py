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


def finding_charts(target, observer, dates, surveys=['SDSSr', 'DSS Red'],
                   text=None, size=15, step=1, lstep=6, lformat='%H:%M',
                   alpha=0.5, lalpha=1, **kwargs):
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
      The target designation.
    observer : string
      The observer location.
    date : array-like of string
      The start and end dates for the finding charts.  Processed with
      `util.date2time`.
    surveys : array-like, optional
      List of surveys to display in order of preference.  See
      `astroquery.skyview`.
    size : int, optional
      Field of view in arcmin.
    test : string, optiona
      Additional text to add to the upper-left corner of the plot.
    step : float, optional
      Length of each time step. [hr]
    lstep : float, optional
      Length of time steps between labels. [hr]
    lformat : string, optional
      Label format, using strftime format codes.
    alpha : float, optional
      Transparency for figure annotations.  0 to 1 for transparent to
      solid.
    lalpha : float, optional
      Transparency for labels.
    **kwargs
      Additional keyword arguments are passed to astroquery's
      `~Horizons.ephemerides`.

    """

    import os
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import skycoord_to_pixel
    import astropy.units as u
    import astropy.coordinates as coords
    from astroquery.skyview import SkyView
    from astroquery.jplhorizons import Horizons
    import aplpy
    from .. import util

    # dates
    start, stop = util.date2time(dates)

    jd = np.arange(start.jd, stop.jd + step / 24, step / 24)
    q = Horizons(id=target, id_type='designation', location=observer,
                 epochs=jd)

    kwargs['closest_apparition'] = kwargs.get('closest_apparition', True)
    kwargs['no_fragments'] = kwargs.get('no_fragments', True)
    tab = q.ephemerides(**kwargs)
    eph = coords.SkyCoord(u.Quantity(tab['RA']), u.Quantity(tab['DEC']))

    jd_labels = np.arange(start.jd, stop.jd + lstep / 24, lstep / 24)
    q.epochs = jd_labels
    tab = q.ephemerides(**kwargs)
    labels = coords.SkyCoord(u.Quantity(tab['RA']), u.Quantity(tab['DEC']))

    step = 0
    while step < len(eph):
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
            d = util.date2time(jd_labels[k])
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
        d = util.date2time(jd[step])
        d = d.isot[:16].replace('-', '').replace(':', '').replace('T', '_')
        fig.save('{}-{}-{}.png'.format(t, d, fn),
                 dpi=300)

        step = np.flatnonzero(i)[-1] + 1


if __name__ == "__main__":
    import argparse
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
    parser.add_argument('--cap', action='store_true',
                        help='For comets, request the Closest APparition from HORIZONS.')
    parser.add_argument('--nofrag', action='store_true',
                        help='For comets, disable HORIZONS nucleus fragment matching.')

    args = parser.parse_args()
    target = ' '.join(args.target)
    assert len(target.strip()) > 0

    finding_charts(target, args.observer, [args.start, args.end],
                   step=args.step, lstep=args.lstep, lformat=args.lformat,
                   alpha=args.alpha, closest_apparition=args.cap,
                   no_fragments=args.nofrag)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
