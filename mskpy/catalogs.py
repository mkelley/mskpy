# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
catalogs - Tools for working with lists of stars
================================================

.. autosummary::
   :toctree: generated/

   brightest
   faintest
   find_offset
   project_catalog
   nearest_match
   triangles
   triangle_match

"""

__all__ = [
    'brightest',
    'faintest',
    'find_offset',
    'project_catalog',
    'nearest_match',
    'triangles',
    'triangle_match',
]

import numpy as np

def brightest(cat0, flux, n, full_output=False):
    """Return the n brightest objects in the catalog.

    Parameters
    ----------
    cat0 : array
      2xN array of positions.
    flux : array
      N-element array of object brightness.
    n : int
      Return the brightest `n` objects.
    full_output : bool, optional
      Return optional output.

    Returns
    -------
    cat : ndarray
    i : ndarray

    """
    i = np.argsort(flux)[::-1][:n]
    if full_output:
        return np.array(cat0)[:, i], i
    else:
        return np.array(cat0)[:, i]

def faintest(cat0, flux, n, full_output=False):
    """Return the n faintest objects in the catalog.

    Parameters
    ----------
    cat0 : array
      2xN array of positions.
    flux : array
      N-element array of object brightness.
    n : int
      Return the brightest `n` objects.
    full_output : bool, optional
      Return optional output.

    Returns
    -------
    cat : ndarray
    i : ndarray

    """
    i = np.argsort(flux)[:n]
    if full_output:
        return np.array(cat0)[:, i], i
    else:
        return np.array(cat0)[:, i]

def find_offset(cat0, cat1, matches, tol=3.0, mc_thresh=15):
    """Find the offset between two catalogs, given matched stars.

    The matched star list may have false matches.

    Parameters
    ----------
    cat0, cat1 : array
      2xN array of positions.
    matches : dict
      The best match for star `i` of `cat0` is `matches[i]` in `cat1`.
    tol : float
      The distance tolerance.
    mc_thresh : int
      Process with `meanclip` when the number of points remaining
      after histogram clipping is `>=mc_thresh`.

    Returns
    -------
    dy, dx : float

    """

    from .util import midstep, meanclip

    d = cat0[:, matches.keys()] - cat1[:, matches.values()]
    bins = (np.arange(d[0].min() - 2 * tol, d[0].max() + 2 * tol, 2 * tol),
            np.arange(d[1].min() - 2 * tol, d[1].max() + 2 * tol, 2 * tol))
    h, edges = np.histogramdd(d.T, bins=bins)

    i = np.unravel_index(h.argmax(), h.shape)
    peak = midstep(edges[0])[i[0]], midstep(edges[1])[i[1]]

    i = np.prod(np.abs(d.T - peak) < tol, 1, dtype=bool)
    good = d[:, i]

    if i.sum() >= mc_thresh:
        j = meanclip(d[0, i], full_output=True)[2]
        k = meanclip(d[1, i], full_output=True)[2]
        good = good[:, list(set(np.r_[j, k]))]

    return good.mean(1)

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

def nearest_match(cat0, cat1, tol=1.0, **kwargs):
    """Find nearest neighbors in two lists.

    Parameters
    ----------
    cat0, cat1 : arrays
      Each catalog is a 2xN array of (y, x) positions.
    tol : float, optional
      The radial match tolerance in pixels.
    **kwargs
      Keyword arguments for `meanclip` when computing mean offsets.

    Returns
    -------
    matches : dictionary
      The best match for star `i` of `cat0` is `matches[i]` in `cat1`.
      Stars that are matched multiple times, or not matched at all,
      are not returned.
    dyx : ndarray
      Sigma-clipped mean offset.  Clipping is done in x and y
      separately, and only the union of the two is returned.

    """

    from scipy.spatial.ckdtree import cKDTree
    from .util import takefrom, meanclip

    assert len(cat0) == 2
    assert len(cat1) == 2

    tree = cKDTree(cat0.T)
    d, i = tree.query(cat1.T)  # 0 of cat1 -> i[0] of cat0

    matches = dict()
    for k, j in enumerate(i):
        if d[k] < tol:
            matches[j] = k

    dyx = cat0[:, matches.keys()] - cat1[:, matches.values()]
    mcx = meanclip(dyx[1], full_output=True)
    mcy = meanclip(dyx[0], full_output=True)
    j = list(set(np.concatenate((mcx[2], mcy[2]))))

    return matches, dyx[:, j]

def triangles(y, x, max_ratio=100, min_sep=0):
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
      Maximum ratio of longest to shortest side.  ccxymatch claims 5
      to 10 is best, but I found better results with 100.
    min_sep : float
      The minimum distance between two vertices.

    Returns
    -------
    v : array
      Indices of the `y` and `x` arrays which define the vertices of
      each triangle (3xN).  The first vertex is opposite side a, the
      second, b, and the third, c.
    s : array
      The shape of each triangle (4xN): perimeter, a/c, cos(beta),
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

    return v.T, shapes.T

def triangle_match(cat0, cat1, tol=0.01, a2c_tol=1.0, cbet_tol=0.2,
                   psig=1.0, pscale=None, min_frac=0, msig=None,
                   full_output=False, verbose=True, **kwargs):
    """Find spatially matching sources between two lists.

    Following IRAF's ccxymatch.  Compare the shapes and orientations
    of triangles within two catalog lists:

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
      The triangle match tolerance in (a/c, cos(beta)) space.
    a2c_tol : float, optional
      A tolerance factor for the a/c ratio matching.
    cbet_tol : float, optional
      A tolerance factor for the cos(beta) matching.
    psig : float, optional
      Clip perimeter ratios at this sigma.
    pscale : float, optional
      If set, only consider perimeter ratios near `pscale`.
    min_frac : float, optional
      Only return stars with `frac > min_frac`.
    msig : float, optional
      Only return stars with matches better than `msig` above the
      match_matrix background.
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
      The scores for each triangle's nearest neighbor, or the radial
      distance for each nearest-neighbor.

    """

    from scipy.spatial.ckdtree import cKDTree
    from .util import takefrom, meanclip

    msig = -100 if msig is None else msig

    assert len(cat0) == 2
    v0, s0 = triangles(*cat0, **kwargs)

    assert len(cat1) == 2
    v1, s1 = triangles(*cat1, **kwargs)

    N0 = len(cat0[0])
    N1 = len(cat1[0])

    # Find closest triangles, considering a/c and cos(beta)
    shape_scale = 1.0 / np.array((a2c_tol, cbet_tol))
    tree = cKDTree(s0[1:3].T * shape_scale)
    d, i = tree.query(s1[1:3].T * shape_scale)

    # reject based on tolerance
    good = d <= tol

    if verbose:
        print(("""[triangle_match] cat0 = {} triangles, cat1 = {} triangles
[triangle_match] Best match score = {:.2g}, worst match sorce = {:.2g}
[triangle_match] {} triangle pairs at or below given tolerance ({})""").format(
                v0.shape[1], v1.shape[1], min(d), max(d), sum(d <= tol), tol))

    # reject based on perimeter
    perimeter_ratios = s1[0] / s0[0, i]
    if pscale is not None:
        a = good.sum()
        for j in range(3):
            sig = np.sqrt(np.mean((perimeter_ratios[good] - pscale)**2))
            good *= np.abs(perimeter_ratios - pscale) < sig * psig
        b = good.sum()
        if verbose:
            print(("""[triangle_match] One time perimeter_ratio sigma clip about {}
[triangle_match] {} triangles rejected.""").format(pscale, a - b))


    mc = meanclip(perimeter_ratios[good], lsig=psig, hsig=psig,
                  full_output=True)
            
    if mc[1] <= 0:
        print("[triangle_match] Low measured perimeter ratio sigma ({}), skipping perimeter rejection".format(mc[1]))
    else:
        p_good = (np.abs(perimeter_ratios - mc[0]) / mc[1] <= psig) * good
        good *= p_good

        if verbose:
            print(("""[triangle_match] Sigma-clipped perimeter ratios = {} +/- {}
[triangle_match] {} triangle pairs with perimeter ratios within {} sigma."""
               ).format(mc[0], mc[1], p_good.sum(), psig))

    # reject based on orientation of vertices
    ccw = s0[3, i] == s1[3]
    cw_count = (~ccw[good]).sum()
    ccw_count = ccw[good].sum()
    if ccw_count >= cw_count:
        good *= ccw
    else:
        good *= ~ccw

    if verbose:
        print(("""[triangle_match] Orientation matches = {}{}
[triangle_match] Anti-orientation matches = {}{}""").format(
    cw_count, " (rejected)" if cw_count <= ccw_count else "",
    ccw_count, " (rejected)" if cw_count > ccw_count else ""))

    match_matrix = np.zeros((N0, N1), int)
    for k, j in enumerate(i):
        if good[k]:
            match_matrix[v0[0][j], v1[0][k]] += 1
            match_matrix[v0[1][j], v1[1][k]] += 1
            match_matrix[v0[2][j], v1[2][k]] += 1

    mc = meanclip(match_matrix, full_output=True)
    m0 = match_matrix.argmax(1)
    m1 = match_matrix.argmax(0)
    matches = dict()
    frac = dict()
    rej = 0
    for i in range(len(m0)):
        if i == m1[m0[i]]:
            if match_matrix[i, m0[i]] < (mc[0] + msig * mc[1]):
                rej += 1
                continue
            matches[i] = m0[i]
            peak = match_matrix[i, m0[i]] * 2
            total = match_matrix[i, :].sum() + match_matrix[:, m0[i]].sum()
            frac[i] = peak / float(total)

    if verbose:
        print("[triangle_match] {} stars failed the msig test".format(rej))

    for k in matches.keys():
        if frac[k] < min_frac:
            del matches[k], frac[k]

    if verbose:
        print("[triangle_match] {} stars matched".format(len(matches)))

    if full_output:
        return matches, frac, match_matrix, d
    else:
        return matches, frac

