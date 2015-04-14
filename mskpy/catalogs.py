# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
catalogs - Tools for working with lists of stars
================================================

.. autosummary::
   :toctree: generated/

   brightest
   project_catalog
   spatial_match
   triangles

"""

__all__ = [
    'brightest',
    'project_catalog',
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

def brightest(cat0, flux, n):
    """Return the n brightest objects in the catalog.

    Parameters
    ----------
    cat0 : array
      2xN array of positions.
    flux : array
      N-element array of object brightness.
    n : int
      Return the brightest `n` objects.

    Returns
    -------
    cat : array

    """
    return cat0[:, np.argsort(flux)[::-1][:n]]

def spatial_match(cat0, cat1, tol=0.1, a2c_tol=1.0, cbet_tol=0.2,
                  psig=1.5, min_frac=0, full_output=False, verbose=True,
                  **kwargs):

    """Find spatially matching sources between two lists.

    Following IRAF's ccxymatch, compare the shapes and orientations of
    triangles within two catalog lists:

      1) For each triangle, find the nearest neighbor in `(a/c,
      cos(beta))` space, where `a` and `c` are the longest and shorest
      sides, respectively, and `beta` is the angle between the
      longtest and shortest side (i.e., the opening angle of the
      vertex opposite side `b`).

      2) If the distance between two triangles in `(a/c, cos(beta))`
      space is `d`, then ignore triangles with `d < tol`.

      3) Compute the ratio of the matched triangle perimeters, and
      reject poor matches based on a sigma-clipping algorithm.  This
      step essentially finds the scale factor between the two
      catalogs.

      4) Compare matched triangle orientations, and compute the number
      that match in the clockwise sense.  If this is more than half,
      then reject those that match in the counter-clockwise sense,
      otherwise, reject the clockwise matches.

      5) Finally, for each triangle find the actual stars that match.
      Optionally, remove stars that do not match a given fraction of
      the time.

    Parameters
    ----------
    cat0, cat1 : arrays
      Each catalog is a 2xN array of (y, x) positions.
    tol : float, optional
      The match tolerance in (a/c, cos(beta)) space.
    a2c_tol : float, optional
      A tolerance factor for the a/c ratio matching.
    cbet_tol : float, optional
      A tolerance factor for the cos(beta) matching.
    psig : float, optional
      Clip perimeter ratios at this sigma.
    min_frac : float, optional
      Only return stars with `frac > min_frac`.
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
    frac : dictionary
      Fraction of times star `i` matched star `matches[i]` out of all
      times stars `i` and `matches[i]` were matched to any star.
    match_matrix : ndarray, optional
      If `full_output` is `True`, also return the matrix of all star
      matches.
    scores : ndarray, optional
      The scores for each triangle's nearest neighbor.

    Notes
    -----

    Step 5 is not robust against vertex re-ordering.  For example, if
    triangle a, b, c matches d, e, f, but compared with triangle e, f,
    d, then the match will not be as good.

    """

    from scipy.spatial.ckdtree import cKDTree
    from .util import takefrom, meanclip

    print "\n*** spatial_match is experimental! See Notes. ***\n"

    assert len(cat0) == 2
    v0, s0 = triangles(*cat0, **kwargs)

    assert len(cat1) == 2
    v1, s1 = triangles(*cat1, **kwargs)

    N0 = len(cat0[0])
    N1 = len(cat1[0])

    # Find closest triangles, considering a/c and cos(beta)
    scale = 1.0 / np.array((a2c_tol, cbet_tol))
    tree = cKDTree(s0[:, 1:3] * scale)
    d, i = tree.query(s1[:, 1:3] * scale)

    # reject based on tolerance
    good = d <= tol

    if verbose:
        print ("""[spatial_match] cat0 = {} triangles, cat1 = {} triangles
[spatial_match] Best match score = {:.2g}, worst match sorce = {:.2g}
[spatial_match] {} triangle pairs at or below given tolerance ({})""").format(
                len(v0), len(v1), min(d), max(d), sum(d <= tol), tol)

    # reject based on perimeter
    perimeter_ratios = s1[:, 0] / s0[i, 0]
    mc = meanclip(perimeter_ratios[good], lsig=psig, hsig=psig,
                  full_output=True)
    if mc[1] <= 0:
        print "[spatial_match] Low measured perimeter ratio sigma ({}), skipping perimeter rejection".format(mc[1])
    else:
        p_good = (np.abs(perimeter_ratios - mc[0]) / mc[1] <= psig) * good
        good *= p_good

        if verbose:
            print ("""[spatial_match] Sigma-clipped perimeter ratios = {} +/- {}
[spatial_match] {} triangle pairs with perimeter ratios within {} sigma."""
               ).format(mc[0], mc[1], p_good.sum(), psig)

    # reject based on orientation of vertices
    ccw = s0[i, 3] == s1[:, 3]
    cw_count = (~ccw[good]).sum()
    ccw_count = ccw[good].sum()
    if ccw_count >= cw_count:
        good *= ccw
    else:
        good *= ~ccw

    if verbose:
        print ("""[spatial_match] Orientation matches = {}{}
[spatial_match] Anti-orientation matches = {}{}""").format(
    cw_count, " (rejected)" if cw_count <= ccw_count else "",
    ccw_count, " (rejected)" if cw_count > ccw_count else "")

    match_matrix = np.zeros((N0, N1), int)
    for k, j in enumerate(i):
        if good[k]:
            match_matrix[v0[j][0], v1[k][0]] += 1
            match_matrix[v0[j][1], v1[k][1]] += 1
            match_matrix[v0[j][2], v1[k][2]] += 1

    m0 = match_matrix.argmax(1)
    m1 = match_matrix.argmax(0)
    matches = dict()
    frac = dict()
    for i in range(len(m0)):
        if i == m1[m0[i]]:
            matches[i] = m0[i]
            peak = match_matrix[i, m0[i]] * 2
            total = match_matrix[i, :].sum() + match_matrix[:, m0[i]].sum()
            frac[i] = peak / float(total)

    for k in matches.keys():
        if frac[k] < min_frac:
            del matches[k], frac[k]

    if full_output:
        return matches, frac, match_matrix, d
    else:
        return matches, frac

def triangles(y, x, max_ratio=10, min_sep=0):
    """Describe all possible triangles in a set of points.

    Following IRAF's ccxymatch, triangles computes:

      * length of the perimeter
      * ratio of longest to shortest side
      * cosine of the angle between the longest and shortest side
      * the direction of the arrangement of the vertices of each
        triangle

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
      each triangle (Nx3).  The first vertex is opposite side a, the
      second, b, and the third, c.
    s : array
      The shape of each triangle (Nx4): perimeter, a/c, cos(beta),
      orientation.

    """

    from itertools import combinations

    v = np.array(list(combinations(range(len(y)), 3)))

    dy = y[v] - np.roll(y[v], 1, 1)
    dx = x[v] - np.roll(x[v], 1, 1)
    sides = np.sqrt(dy**2 + dx**2)

    # numpy magic from
    # http://stackoverflow.com/questions/10921893/numpy-sorting-a-multidimensional-array-by-a-multidimensional-array/
    i = np.argsort(sides, 1)[:, ::-1]  # indices of sides a, b, c
    i = list(np.ogrid[[slice(j) for j in i.shape]][:-1]) + [i]
    v = v[i]
    dy = dy[i]
    dx = dx[i]
    abc = sides[i]

    a2c = abc[:, 0] / abc[:, 2]
    i = (a2c < max_ratio) * (abc[:, 2] > min_sep)
    dx = dx[i]
    dy = dy[i]
    abc = abc[i]
    a2c = a2c[i]

    perimeter = abc.sum(1)
    cbet = ((abc[:, 0]**2 + abc[:, 2]**2 - abc[:, 1]**2)
            / (2 * abc[:, 0] * abc[:, 2]))
    rot = np.sign((x[v[:, 0]] - x[v[:, 2]]) * (y[v[:, 1]] - y[v[:, 0]])
                  - (x[v[:, 0]] - x[v[:, 1]]) * (y[v[:, 2]] - y[v[:, 0]]))
    rot = rot[i]
    shapes = np.c_[perimeter, a2c, cbet, rot]

    return v, shapes
