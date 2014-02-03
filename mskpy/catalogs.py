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

def spatial_match(cat0, cat1, tol=0.01, full_output=False, verbose=True):
    """Find spatially matching sourse between two lists.

    Parameters
    ----------
    cat0, cat1 : arrays
      Each catalog is an 2xN array of (y, x) positions.
    tol : float, optional
      The match tolerance.
    full_output : bool, optional
      Set to `True` to also return `match_matrix`.
    verbose : bool, optional
      Print some feedback for the user.

    Returns
    -------
    matches : dictionary
      The best match for star `i` of `cat0` is `matches[i]` in `cat1`.
      Stars that are matched multiple times, or not matched at all,
      are not returned.
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

    v0, s0 = triangles(*cat0)
    v1, s1 = triangles(*cat1)
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

    if full_output:
        return matches, scores, match_matrix
    else:
        return matches, scores

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
    dy = y[v] - np.roll(y[v], 1, 1)
    dx = x[v] - np.roll(x[v], 1, 1)
    abc = np.sort(np.sqrt(dy**2 + dx**2), 1)[:, ::-1]  #  lengths of sides abc
    shapes = abc[:, 1:] / abc[:, :1]  # b/a, c/a

    return v, shapes
