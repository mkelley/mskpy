# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
catalogs - Tools for working with lists of stars
================================================

.. autosummary::
   :toctree: generated/

   match_cat
   triangles

"""

__all__ = [
    'match_cat',
    'triangles'
]

import numpy as np

def match_cat(cat0, cat1, tol=0.01, verbose=True):
    """Find matching stars between two catalogs.

    Parameters
    ----------
    cat0, cat1 : arrays
      Each catalog is an 2xN array of (y, x) positions.
    tol : float, optional
      The match tolerance.
    verbose : bool, optional
      Print some feedback for the user.

    Returns
    -------
    matches : ndarray
      The best match for star `i` of `cat0` is index `matches[i]` in
      `cat1`.
    confidence : ndarray
      The number of times star `i` of `cat0` was matched to
      `matches[i]` of `cat1`.

    Notes
    -----

    Based on the description of DAOPHOT's catalog matching via
    triagles at
    http://ned.ipac.caltech.edu/level5/Stetson/Stetson5_2.html

    """

    from scipy.spatial.ckdtree import cKDTree

    v0, s0 = triangles(*cat0)
    v1, s1 = triangles(*cat1)
    tree = cKDTree(s0)
    d, i = tree.query(s1)  # nearest matches between triangles

    if verbose:
        print ("""[match_cat] cat0 = {} triangles, cat1 = {} triangles
[match_cat] Best match score = {}, worst match sorce = {}
[match_cat] {} matches at or below given tolerance ({})""".format(
                len(v0), len(v1), min(d), max(d), sum(d <= tol), tol))
    
    matched = np.zeros((len(cat0[0]), len(cat1[0])), int)
    for k, j in enumerate(i):
        if d[k] <= tol:
            matched[v0[j][0], v1[k][0]] += 1
            matched[v0[j][1], v1[k][1]] += 1
            matched[v0[j][2], v1[k][2]] += 1

    matches = np.argmax(matched, 1)
    confidence = np.max(matched, 1)
    return matches, confidence

def triangles(y, x):
    """Describe all possible triangles in a set of points.

    Parameters
    ----------
    y, x : arrays
      Lists of coordinates.

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
    dy = y[v] - np.roll(y[v], 1)
    dx = x[v] - np.roll(x[v], 1)
    abc = np.sort(np.sqrt(dy**2 + dx**2), 1)[:, ::-1]  #  lengths of sides abc
    shapes = abc[:, 1:] / abc[:, :1]  # b/a, c/a

    return v, shapes
