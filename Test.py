import numpy as np
import os
import datetime
import numpy.ma as ma
import time
import random
from scipy.optimize import curve_fit
from numba import jit

########################################################################################################################
#  Non-gridded IPA.npy
#
#  This script runs the non-gridded 2D isotropic inner product analysis for Seasons in
#  Seasonlist, with N percentage of observations. Prior to this script running there are
#  files created in \\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\ and these are
#  required for each season run. This includes .npz files for observations,
#  coarse_grid_sst.npz (for the chosen seasons), and a rossby radius .npy file.
#
########################################################################################################################

KM2D = 40000/360
BinSize = 30
MaxKM = 900
MaxRadius = MaxKM/KM2D
dd = (BinSize / KM2D)
M = int(MaxKM/BinSize)+1
dist = np.arange(M) * dd


@jit('Tuple((f8, f8, f8))(f8[:], f8[:], f8[:], i8)', nopython=True, parallel=True, forceobj=True)
def time_loop(xti, yti, zti, daycount):

    fitcov, fitcovcount = np.zeros(M), np.zeros(M)
    popt, pcov = np.nan, np.nan
    for tt in range(0, daycount):
        # Set array of 'today's' innovations
        xxi = xti[tt]
        yyi = yti[tt]
        zzi = zti[tt]
        rr = np.sqrt(xxi*xxi + yyi*yyi)
        if len(rr) <= 3:
            continue
        fitcov, fitcovcount = binning(rr, zzi, fitcov, fitcovcount)
    fitcovcount[fitcovcount == 0] = np.nan
    cov = fitcov/fitcovcount

    if (cov != 0.0).any():
        valididx = np.logical_not(np.isnan(cov))
        valididx[0] = False

        popt, pcov = curve_fit(func, dist[valididx], cov[valididx], maxfev=1000)
    return popt[0], popt[1], cov[0]


@jit('Tuple((f8[:], f8[:]))(f8[:], f4[:], f8[:], f8[:])', nopython=True, parallel=True, fastmath=True)
def binning(rr, zzi, fc, fcc,):
    v0idx = np.argmin(rr)
    v0 = zzi[v0idx]
    idd = rr < MaxRadius
    f = v0 * zzi[idd]
    rr = rr[idd]
    for p, q in enumerate(f):
        fc[int(rr[p] / dd)] += q
        fcc[int(rr[p] / dd)] += 1

    return fc, fcc


#  Creating the coarse grid for storing the innovations and producing output. In both 1d and 2d arrays.

stepy = int((32 - 8) / 0.274)
yedges = np.linspace(8, 32, int(stepy))
stepx = int((75 - 45) / 0.3)
xedges = np.linspace(45, 75, int(stepx))
ycenter = 0.5 * (yedges[1:] + yedges[:-1])
xcenter = 0.5 * (xedges[1:] + xedges[:-1])

nlist = [100/100, 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]
Seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]  # "1-DJF", "2-MAM", "3-JJA", "4-SON"
Overwrite = True
a1 = 4  # Long length-scale, in degrees.
Typ = 'sst'

size = []
TIME = []
dt = datetime.timedelta(hours=24)
for N in nlist:  # Use every Nth observation. (N=1 is every observation)
    for Season in Seasonlist:
        S1 = time.time()
        Zi = []
        Yi = []
        Xi = []

        if Season == '1-DJF':
            Start = datetime.date(2013, 12, 1)
            End = datetime.date(2014, 3, 1)
        elif Season == '2-MAM':
            Start = datetime.date(2014, 3, 1)
            End = datetime.date(2014, 6, 1)
        elif Season == '3-JJA':
            Start = datetime.date(2014, 6, 1)
            End = datetime.date(2014, 9, 1)
        elif Season == '4-SON':
            Start = datetime.date(2014, 9, 1)
            End = datetime.date(2014, 12, 1)
        else:
            End = 0
            Start = 0
            exit(0)

        dayCount = (End - Start).days
        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\coarse_grid_%s' % Typ)

        random.seed(100)
        ze = []
        if Overwrite is True:
            for t in range(dayCount):
                date = Start + t * dt
                # Open and save all innovations,
                #  Observations chosen as a random 1/N % of the whole data set. Seeded with 100.

                dat = np.load('%i/%04i-%02i-%02i.npz' % (date.year, date.year, date.month, date.day))
                idx = sorted(random.sample(list(np.arange(len(dat['zi'][:, 0]))), k=int(len(dat['zi'][:, 0]) / N)))
                ze.append(len(dat['zi'])/N)
                Zi.append(dat['zi'][idx, 0])
                Xi.append(dat['xi'][idx])
                Yi.append(dat['yi'][idx])
        size.append(np.sum(ze))
        os.chdir('../%s/' % Season)

        # Load coarse grid for domain and mask
        GRID = np.load('coarse_grid_%s.npz' % Typ)
        gridz = GRID['zi']
        gridz = ma.masked_invalid(gridz)
        GRID.close()

        #  Load rossby radius in terms of the coarse grid. Commented sections kept incase of need for recalculation.
        # ross = interpolate.griddata((rlon, rlat), np.array(Ross), (x2, y2), 'nearest')

        ross = np.load('rossby.npy')
        ross = (ross / 1000) / KM2D

        STD, LSR, obs = np.array(np.zeros_like(gridz)), np.array(np.zeros_like(gridz)), np.array(np.zeros_like(gridz))

        for i, y in enumerate(ycenter):
            if np.sum(gridz.mask[i, :]) == 99:
                continue
            print(i, time.time() - S1)

            xiY = []
            yiY = []
            ziY = []

            for t in range(dayCount):
                idx = (abs(Yi[t] - y) <= MaxRadius)
                xiY.append(Xi[t][idx])
                yiY.append(Yi[t][idx])
                ziY.append(Zi[t][idx])

            for j, x in enumerate(xcenter):
                if np.sum(gridz.mask[i, j]) == 1:
                    continue
                S2 = time.time()

                a0 = ross[i, j]
                xi = []
                yi = []
                zi = []
                for t in range(0, dayCount):  # 'Box Cut', removes all values more than 9 degrees in x or y.
                    idx = (abs(xiY[t] - x) <= MaxRadius)
                    xi.append((xiY[t][idx]-x))
                    yi.append((yiY[t][idx]-y))
                    zi.append(ziY[t][idx])

                def func(xx, xa, xb):  # The function is created anew, as it is dependent on the Rossby radius.
                    return xa * np.exp(-(xx ** 2) / (2 * a0 ** 2)) + xb * np.exp(-(xx ** 2) / (2 * a1 ** 2))
                a, b, V = time_loop(xi, yi, zi, dayCount)
                # print(time.time()-S2)
                af = a/(a+b)
                mf = a+b
                STD[i, j] = mf
                LSR[i, j] = af
                obs[i, j] = V - mf
        LSR[LSR == 0] = np.nan
        STD[STD == 0] = np.nan
        obs[obs == 0] = np.nan

        print(time.time()-S1)
        TIME.append(time.time()-S1)
        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s\%s' % (Typ.upper(), Season))
        if os.path.isdir('%i/test' % (100/N)) is False:
            os.makedirs('%i/test' % (100/N))
        np.save(r"%i\test\Data\IPA_Sdv_Real.npy" % (100/N), STD)
        np.save(r"%i\test\Data\IPA_Obs_Real.npy" % (100/N), obs)
        np.save(r"%i\test\Data\IPA_Lsr_Real.npy" % (100/N), LSR)

os.chdir('../')
txt = open('IPA Diagnostics-test.txt', 'w+')
txt.write('    Time    |    N     |    Season    |    Obs    \n')
m = 0
for N in nlist:
    for p2 in range(len(Seasonlist)):
        txt.write('  %7.2f   |  %5.2f   |     %s    |    %i\n' % (
            TIME[p2+m], 1/N, Seasonlist[p2], int(size[p2+m])))
    m += len(Seasonlist)
txt.close()
