import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from mskpy import photometry

phot = ascii.read('''
  # standard star photometry
  # m : apparent magnitude (Farnham et al. 2000)
  # m_inst : instrumental magnitude
  # z : zenith angle in degrees
  #
  filter   m    m_inst m_inst unc   z    airmass
  ------ ----- ------- ---------- ------ -------
      RC 7.766 -13.877      0.004 59.932   1.989
      RC 7.766 -13.902      0.003 48.062   1.494
      RC 7.448 -14.236      0.001 42.090   1.346
      RC 7.448 -14.211      0.003 42.553   1.356
      BC  7.68 -14.482      0.007 59.565   1.968
      BC  7.68 -14.604      0.006 47.767   1.486
      BC 7.784 -14.518      0.005 42.690   1.359
      BC 7.784 -14.543      0.005 41.977   1.344
      CN 7.619 -13.936      0.008 60.297   2.011
      CN 7.619 -14.137      0.007 48.360   1.503
      CN 7.748 -14.096      0.003 42.210   1.349
      CN 7.748 -14.071      0.006 42.422   1.353
      OH 7.414  -8.746      0.037 60.691   2.036
      OH 7.414  -9.731      0.014 48.685   1.512
      OH 7.536  -9.955      0.009 42.339   1.351
      OH 7.536  -9.942      0.012 42.289   1.351
  ''')

h = 2361 * u.m  # elevation of LDT

# first, calibrate the easy filters
cal = []
for filt in ('CN', 'BC', 'RC'):
    data = phot[phot['filter'] == filt]

    # calibrate magnitude with extinction proportional to airmass
    fit, fit_unc = photometry.cal_airmass(
        data['m_inst'], data['m_inst unc'], data['m'], data['airmass'])

    # best fit for each data point:
    model = (data['m'] - fit[0] + fit[1] * data['airmass'])

    # residuals from best fit:
    residuals = model - data['m_inst']

    # save results
    # filter, N, magzp, magzp unc, Ex, Ex unc, toz, toz unc, mean residuals
    # stdev residuals, standard error
    cal.append((
        filt,
        len(data),
        fit[0],
        fit_unc[0],
        fit[1],
        fit_unc[1],
        np.ma.masked,  # ozone parameter not calculated
        np.ma.masked,
        np.mean(residuals),
        np.std(residuals, ddof=1),
        np.std(residuals, ddof=1) / np.sqrt(len(data))
    ))

# OH extinction has multiple components, here we use BC to help isolate the ozone component
data = phot[phot['filter'] == 'OH']
bc = cal[1]
Ex_BC = cal[1][4]

# see cal_oh for explanation of parameters and return values
fit, fit_unc = photometry.hb.cal_oh(
    data['m_inst'], data['m_inst unc'], data['m'], data['z'] * u.deg,
    'b', 'b', Ex_BC, h)

# use best-fit ozone parameter and BC extinction to calculate total extinction in OH
ext_oh = photometry.hb.ext_total_oh(fit[1], data['z'] * u.deg, 'b', 'b',
                                    Ex_BC, h)
model = data['m'] - fit[0] + ext_oh
residuals = model - data['m_inst']

# save results
cal.append((
    'OH',
    len(data),
    fit[0],
    fit_unc[0],
    np.ma.masked,  # not calculated for OH
    np.ma.masked,
    fit[1],
    fit_unc[1],
    np.mean(residuals),
    np.std(residuals, ddof=1),
    np.std(residuals, ddof=1) / np.sqrt(len(data))
))

cal = Table(
    rows=cal,
    names=['filter', 'N', 'magzp', 'magzp unc', 'Ex', 'Ex unc',
           'toz', 'toz unc', 'mean residuals', 'stdev residuals',
           'standard error'])

for col in cal.colnames[2:]:
    cal[col].format = '{:.3f}'

cal.pprint_all()
