# Analysis scripts for 3I/ATLAS JWST nucleus extraction

These scripts were used in the analysis of JWST NIRCam and MIRI observations of 3I/ATLAS to estimate the brightness of the comet's nucleus.  See the paper for an outline of the technique.

1. Get images from MAST.  These scripts use the *_cal.fits files, i.e., not the i2d (distorition corrected) files.
2. Measure centers or use the provided center files.
3. Generate PSFs.  Example scripts are provided.  Note the MIRI data used separate PSFs for the coma and nucleus.
4. Edit parameters in the fit script as needed.
5. Fit the data.

The scripts were highly tailored to these specific observations, but much of the code could be re-used for other data.

Mike Kelley
University of Maryland
2026 July 20
