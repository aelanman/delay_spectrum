# coding: utf-8

import matplotlib
#matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as pl
from  matplotlib.ticker import FuncFormatter

longbl = False
shortbl = not longbl

if shortbl:
    fh = np.load('results/hera_3short_3bls_nside128_24.00hrs_100skies.npz')
    fm = np.load('results/mwa_3short_3bls_nside128_24.00hrs_100skies.npz')
    label = 'short'

if longbl:
    fh = np.load('results/hera_3long_3bls_nside128_24.00hrs_100skies.npz')
    fm = np.load('results/mwa_3long_3bls_nside128_24.00hrs_100skies.npz')
    label= 'long'

#fh = np.load('hera_3short3bls-avg_nside128_24.00hrs_100skies.npz')
#fm = np.load('mwa_3short3bls-avg_nside128_24.00hrs_100skies.npz')

#fh = np.load('hera_3long3bls-avg_nside128_24.00hrs_100skies.npz')
#fm = np.load('mwa_3long3bls-avg_nside128_24.00hrs_100skies.npz')

hrms = np.mean(fh['variances'], axis=2)
mrms = np.mean(fm['variances'], axis=2)
#times = fh['avg_lens_time']
times = fh['avg_lens_time'].astype(float)
ref = fh['ref_line']
herrs = fh['stderrs']
merrs = fm['stderrs']
bllabels = ['{}_{}'.format(b[0], b[1]) for b in fh['bls']]
Nbls = len(bllabels)


if len(hrms.shape) == 1:
    hrms = hrms[:,np.newaxis]
    mrms = mrms[:,np.newaxis]
    herrs = herrs[:,np.newaxis]
    merrs = merrs[:,np.newaxis]
    Nbls = 1
    bllabels=['Avg']

cmap = pl.get_cmap('tab10')
(w,h) = pl.figaspect(9/16.)
fig, ax = pl.subplots(1,1, figsize=(w,h))
#cmap=lambda x : 'k'
for bi, bl in enumerate(bllabels):
    col = cmap(bi)
    pl.fill_between(times, (hrms[:,bi]-herrs[:,bi])/ref, (hrms[:,bi]+herrs[:,bi])/ref, color=col, hatch='-', alpha=0.15, edgecolor=col)
    pl.plot(times, hrms[:,bi]/ref, linestyle='dashed', color=col)
    pl.fill_between(times, (mrms[:,bi]-merrs[:,bi])/ref, (mrms[:,bi]+merrs[:,bi])/ref, color=col, alpha=0.15)
    pl.plot(times, mrms[:,bi]/ref, color=col)

ax.set_xlabel("Averaging time (number of integrations)")
ax.set_ylabel(r"RMS error / $P_{theoretical}$")
ax.set_xlim([np.min(times), np.max(times)])

pl.yscale('log')
ax.minorticks_off()

ax.set_yticks(np.arange(0.1, 1.1, 0.1))
ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))

f = lambda m,c: pl.plot([],[], color=c, ls=m)[0]

colhandles = [f('-', cmap(i)) for i in range(Nbls)]
#markhandles = [f(m, "k") for m in ['-','--']]
markhandles = [f(m, "k") for m in ['-']]

#pl.legend(labels=bllabels + ['MWA', 'HERA'], handles=colhandles + markhandles, loc='upper right')
pl.legend(labels=bllabels + ['Avg'], handles=colhandles + markhandles, loc='upper right')
#pl.legend(labels= ['MWA', 'HERA'], handles= markhandles, loc='upper right')
#ax.add_artist(leg2)

# Plot averaged
#if shortbl:
#    fh = np.load('hera_3short3bls-avg_nside128_24.00hrs_100skies.npz')
#    fm = np.load('mwa_3short3bls-avg_nside128_24.00hrs_100skies.npz')
#if longbl:
#    fh = np.load('hera_3long3bls-avg_nside128_24.00hrs_100skies.npz')
#    fm = np.load('mwa_3long3bls-avg_nside128_24.00hrs_100skies.npz')
#hrms = fh['variances']
#mrms = fm['variances']
#herrs = fh['var_stds']
#merrs = fm['var_stds']
#
#col='k'
#pl.fill_between(times, (hrms-herrs)/ref, (hrms+herrs)/ref, color=col, hatch='-', alpha=0.15, edgecolor=col)
#pl.plot(times, hrms/ref, color='k', linestyle='dashed')
#pl.fill_between(times, (mrms-merrs)/ref, (mrms+merrs)/ref, color=col, alpha=0.15)
#pl.plot(times, mrms/ref, color='k')
#
#pl.plot(times, mrms[0]/times, color='r', lw=1.5)

pl.tight_layout()
pl.grid(linestyle='--', color='0.75')
fig.savefig(label+'rmscurves_nsamp.png', format='png', transparent=True)
#pl.yscale('log')
pl.show()

