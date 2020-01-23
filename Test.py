import numpy as np
import os
import datetime
import numpy.ma as ma
import time
import random
from scipy.optimize import curve_fit
from numba import jit, njit

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
MaxRadius = 900/KM2D


@jit('Tuple((f8, f8, f8))(f8[:], f8[:], f8[:], i8)', forceobj=True)
def time_loop(xti, yti, zti, daycount):
    km2d = 40000 / 360
    maxradius = 900 / km2d
    dd = np.radians(30 / km2d)  # Bin size in metres.
    dist = np.arange(31) * dd

    fitcov, fitcovcount = np.zeros(31), np.zeros(31)
    popt, pcov = np.nan, np.nan
    for tt in range(0, daycount):
        # Set array of 'today's' innovations
        xxi = xti[tt]
        yyi = yti[tt]
        zzi = zti[tt]
        if len(zzi) <= 10:
            continue
        rr = np.sqrt(xxi*xxi + yyi*yyi)
        fitcov, fitcovcount = binning(rr, zzi, fitcov, fitcovcount, maxradius, dd)
    fitcovcount[fitcovcount == 0] = np.nan
    cov = fitcov/fitcovcount

    if (cov != 0.0).any():
        valididx = np.logical_not(np.isnan(cov))
        valididx[0] = False

        popt, pcov = curve_fit(func, dist[valididx], cov[valididx], maxfev=1000000000)
    return popt[0], popt[1], cov[0]


@njit('Tuple((f8[:], f8[:]))(f8[:], f4[:], f8[:], f8[:], f8, f8)', parallel=True, fastmath=True)
def binning(rr, zzi, fc, fcc, mr, dd):
    v0idx = np.argmin(rr)
    v0 = zzi[v0idx]
    idd = rr < np.radians(mr)
    f = v0 * zzi[idd]
    rr = rr[idd]
    for p, q in enumerate(f):
        fc[int(rr[p] / dd)] += q
        fcc[int(rr[p] / dd)] += 1

    return fc, fcc

# @njit('Tuple((f8[:], f8[:,:]))(f8[:], f4[:], f4, i8, f8[:,:], f8[:], f8, f8)', parallel=True, fastmath=True)
# def inner(rr, zzi, v0, v0idx, e, tot, i0, i1):
#
#     exp, sm = np.exp, np.sum
#     idd = rr < np.radians(MaxRadius)*np.radians(MaxRadius)
#     idd[v0idx] = False
#
#     ya = exp(-(rr[idd]) / (2 * i0 * i0))
#     yb = exp(-(rr[idd]) / (2 * i1 * i1))
#
#     e[0, 0] += sm(ya * ya)
#     e[1, 0] += sm(ya * yb)
#     e[0, 1] += sm(ya * yb)
#     e[1, 1] += sm(yb * yb)
#
#     tot[0] += v0*sm(ya * zzi[idd])
#     tot[1] += v0*sm(yb * zzi[idd])
#
#     return tot, e


#  Creating the coarse grid for storing the innovations and producing output. In both 1d and 2d arrays.

stepy = int((32 - 8) / 0.274)
yedges = np.linspace(8, 32, int(stepy))
stepx = int((75 - 45) / 0.3)
xedges = np.linspace(45, 75, int(stepx))
ycenter = 0.5 * (yedges[1:] + yedges[:-1])
xcenter = 0.5 * (xedges[1:] + xedges[:-1])

Seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]  # "1-DJF", "2-MAM", "3-JJA", "4-SON"
Overwrite = True
a1 = 4  # Long length-scale, in degrees.
Typ = 'sst'
size = []
TIME = []
dt = datetime.timedelta(hours=24)
nlist = [100/0.1, 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]
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

        ross = np.load('rossby.npy')
        ross = (ross / 1000) / KM2D

        STD, LSR, obs = np.array(np.zeros_like(gridz)), np.array(np.zeros_like(gridz)), np.array(np.zeros_like(gridz))

        for i, y in enumerate(ycenter):
            if np.sum(gridz.mask[i, :]) == 99:
                continue
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
                    xi.append(np.radians(xiY[t][idx]-x))
                    yi.append(np.radians(yiY[t][idx]-y))
                    zi.append(ziY[t][idx])

                def func(xx, xa, xb):  # The function is created anew, as it is dependent on the Rossby radius.
                    return xa * np.exp(-(xx ** 2) / (2 * np.radians(a0) ** 2)) + xb * np.exp(-(xx ** 2) / (2 * np.radians(a1) ** 2))

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
        if os.path.isdir('%i/Data' % (100/N)) is False:
            os.makedirs('%i/Data' % (100/N))
        np.save(r"%i\test\HL_Sdv_Real.npy" % (100/N), STD)
        np.save(r"%i\test\HL_Obs_Real.npy" % (100/N), obs)
        np.save(r"%i\test\HL_Lsr_Real.npy" % (100/N), LSR)

os.chdir('../')
txt = open('test Diagnostics-Real.txt', 'w+')
txt.write('    Time    |    N     |    Season    |    Obs    \n')
m = 0
for N in nlist:
    for p2 in range(len(Seasonlist)):
        txt.write('  %7.2f   |  %5.2f   |     %s    |    %i\n' % (
            TIME[p2+m], 1/N, Seasonlist[p2], int(size[p2+m])))
    m += len(Seasonlist)
txt.close()


