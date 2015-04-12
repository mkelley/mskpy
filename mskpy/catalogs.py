# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
catalogs - Tools for working with lists of stars
================================================

.. autosummary::
   :toctree: generated/

   spatial_match
   triangles

"""

__all__ = [
    'spatial_match',
    'triangles'
]

import numpy as np

def project_catalog(cat, wcs=None):
    """Project a catalog onto the image plane.

    The default is the tangent image plane, but any WCS transformation
    can be used.

    Parameters
    ----------
    cat : astropy SkyCoord, Quantity, or Angle
      The coordinate list.  If a `Quantity` or `Angle`, `cat` must be
      a 2xN array: (lat, long).
    wcs : astropy.wcs.WCS, optional
      The world coordinate system object for transformation.  The
      default assumes the tangent plane centered on the latitude
      and longitude of the catalog.

    Returns
    -------
    flat_cat : ndarray
      A 2xN array of pixel positions, (y, x).

    """

    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord

    if not isinstance(cat, SkyCoord):
        assert len(cat) == 2
        cat = SkyCoord(ra=cat[1], dec=cat[0])

    if wcs is None:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = (0, 0)
        ra0 = (cat.ra.max() - cat.ra.min()) / 2.0 + cat.ra.min()
        dec0 = (cat.dec.max() - cat.dec.min()) / 2.0 + cat.dec.min()
        wcs.wcs.crval = (ra0.value, dec0.value)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cdelt = np.array([-1, 1]) / 3600.0

    x, y = cat.to_pixel(wcs)
    return np.vstack((y, x))

def spatial_match(cat0, cat1, n=30, tol=0.01, min_score=0, full_output=False,
                  verbose=True, **kwargs):

    """Find spatially matching sources between two lists.

    Parameters
    ----------
    cat0, cat1 : arrays
      Each catalog is a 2xN array of (y, x) positions, or a 3xN array
      of positions and flux (y, x, f).
    n : float, optional
      If fluxes are provided in the catalogs, only the brightest `n`
      stars are considered.
    tol : float, optional
      The match tolerance.
    min_score : float, optional
      Only return matches with scores greater than `min_score`.
    full_output : bool, optional
      Set to `True` to also return `match_matrix`.
    verbose : bool, optional
      Print some feedback for the user.
    **kwargs
      `triangles` keyword arguments.

    Returns
    -------
    matches : dictionary
      The best match for star `i` of `cat0` is `matches[i]` in `cat1`.
      Stars that are matched multiple times, not matched at all, or
      with scores less than `min_score` are not returned.
    score : dictionary
      Fraction of times star `i` matched star `matches[i]` out of all
      times stars `i` and `matches[i]` were matched to any star.
    match_matrix : ndarray, optional
      If `full_output` is `True`, also return the matrix of all star
      matches.

    Notes
    -----

    Based on the description of DAOPHOT's catalog matching via
    triangles at
    http://ned.ipac.caltech.edu/level5/Stetson/Stetson5_2.html

    """

    from scipy.spatial.ckdtree import cKDTree

    if len(cat0) == 3:
        cat0 = cat0[:, np.argsort(cat0[2])[::-1][:n]][:2]
    assert len(cat0) == 2
    v0, s0 = triangles(*cat0, **kwargs)

    if len(cat1) == 3:
        cat1 = cat1[:, np.argsort(cat1[2])[::-1][:n]][:2]
    assert len(cat1) == 2
    v1, s1 = triangles(*cat1, **kwargs)

    N0 = len(cat0[0])
    N1 = len(cat1[0])
    tree = cKDTree(s0)
    d, i = tree.query(s1)  # nearest matches between triangles

    if verbose:
        print ("""[spatial_match] cat0 = {} triangles, cat1 = {} triangles
[spatial_match] Best match score = {:.2g}, worst match sorce = {:.2g}
[spatial_match] {} triangle pairs at or below given tolerance ({})""".format(
                len(v0), len(v1), min(d), max(d), sum(d <= tol), tol))

    match_matrix = np.zeros((N0, N1), int)
    for k, j in enumerate(i):
        if d[k] <= tol:
            match_matrix[v0[j][0], v1[k][0]] += 1
            match_matrix[v0[j][1], v1[k][1]] += 1
            match_matrix[v0[j][2], v1[k][2]] += 1

    m0 = match_matrix.argmax(1)
    m1 = match_matrix.argmax(0)
    matches = dict()
    scores = dict()
    for i in range(len(m0)):
        if i == m1[m0[i]]:
            matches[i] = m0[i]
            peak = match_matrix[i, m0[i]] * 2
            total = match_matrix[i, :].sum() + match_matrix[:, m0[i]].sum()
            scores[i] = peak / float(total)

    for k in matches.keys():
        if scores[k] < min_score:
            del matches[k], scores[k]

    if full_output:
        return matches, scores, match_matrix
    else:
        return matches, scores

def triangles(y, x, max_ratio=10, min_sep=0):
    """Describe all possible triangles in a set of points.

    Parameters
    ----------
    y, x : arrays
      Lists of coordinates.
    max_ratio : float
      Maximum ratio of longest to shortest side.  Typically 5 to 10.
    min_sep : float
      The minimum distance between two vertices.

    Returns
    -------
    v : array
      Indices of the `y` and `x` arrays which define the vertices of
      each triangle (Nx3).
    s : array
      The shape of each triangle (Nx2): b/a, c/a

    """

    from itertools import combinations

    v = np.array(list(combinations(range(len(y)), 3)))
    dy = y[v] - np.roll(y[v], 1, 1)
    dx = x[v] - np.roll(x[v], 1, 1)
    abc = np.sort(np.sqrt(dy**2 + dx**2), 1)[:, ::-1]  #  lengths of sides abc
    shapes = np.vstack((abc[:, 1:] / abc[:, :1]))  # b/a, c/a
    i = ((shapes[:, 1] > max_ratio**-1) * (abc[:, 2] > min_sep))

    return v[i], shapes[i]
