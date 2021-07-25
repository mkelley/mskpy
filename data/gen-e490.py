import numpy as np

w, f = np.loadtxt('E490_00a_AM0.txt').T
h1 = np.histogram(w, bins=np.logspace(np.log10(min(w)), np.log10(10.01), 200))
h2 = np.histogram(w, bins=h1[1], weights=f)

i = w > 10.0
f_sm = np.r_[h2[0] / h1[0], f[i]]
w_sm = np.r_[(h2[1][1:] + h2[1][:-1]) / 2.0, w[i]]

j = np.isfinite(f_sm)
e490 = np.c_[w_sm[j], f_sm[j]]

with open('e490-lowres.txt', 'w') as outf:
    outf.write('''# E490, histogrammed up to 10 um.
# wave (um)   flux (W/m2/um)
''')
    np.savetxt(outf, e490)
