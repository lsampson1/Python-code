import numpy as np
import os
import datetime
import numpy.ma as ma
import time
import random
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

MaxRadius = 900/(40000/360)


@jit('Tuple((f8[:], f8[:,:], f8))(f8[:], f8[:], f8[:], f4, i8, i8)', forceobj=True)
def time_loop(xti, yti, zti, i0, i1, daycount):
    tot = np.zeros(2)
    v = np.zeros(2)
    e = np.zeros((2, 2))
    for tt in range(0, daycount):
        # Set array of 'today's' innovations
        xxi = xti[tt]
        yyi = yti[tt]
        zzi = zti[tt]
        if len(zzi) <= 1:
            continue
        rr = xxi*xxi + yyi*yyi
        v0idx = np.argmin(rr)
        v0 = zzi[v0idx]

        tot, e = inner(rr, zzi, v0, v0idx, e, tot, i0, i1)
        v[0] += v0**2
        v[1] += 1
    return tot, e, v[0]/v[1]


@njit('Tuple((f8[:], f8[:,:]))(f8[:], f4[:], f4, i8, f8[:,:], f8[:], f8, f8)', parallel=True, fastmath=True)
def inner(rr, zzi, v0, v0idx, e, tot, i0, i1):

    exp, sm = np.exp, np.sum
    idd = rr < np.radians(MaxRadius)*np.radians(MaxRadius)
    idd[v0idx] = False

    ya = exp(-(rr[idd]) / (2 * i0 * i0))
    yb = exp(-(rr[idd]) / (2 * i1 * i1))

    e[0, 0] += sm(ya * ya)
    e[1, 0] += sm(ya * yb)
    e[0, 1] += sm(ya * yb)
    e[1, 1] += sm(yb * yb)

    tot[0] += v0*sm(ya * zzi[idd])
    tot[1] += v0*sm(yb * zzi[idd])

    return tot, e


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

#  Ross = (Ross/1000)/(40000/360) # m to degrees, according to https://stackoverflow.com/questions/5217348/how-do-i-conv
#  ert-kilometres-to-degrees-in-geodjango-geos

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

        #  Load rossby radius in terms of the coarse grid. Commented sections kept incase of need for recalculation.
        # ross = interpolate.griddata((rlon, rlat), np.array(Ross), (x2, y2), 'nearest')

        ross = np.load('rossby.npy')
        ross = (ross / 1000) / (40000 / 360)

        STD, LSR, obs = np.array(np.zeros_like(gridz)), np.array(np.zeros_like(gridz)), np.array(np.zeros_like(gridz))

        for i, x in enumerate(xcenter):
            if np.sum(gridz.mask[:, i]) == 86:
                continue
            xiX = []
            yiX = []
            ziX = []

            for t in range(dayCount):
                idx = (abs(Xi[t] - x) <= MaxRadius)
                xiX.append(Xi[t][idx])
                yiX.append(Yi[t][idx])
                ziX.append(Zi[t][idx])

            for j, y in enumerate(ycenter):
                if np.sum(gridz.mask[j, i]) == 1:
                    continue
                S2 = time.time()

                a0 = ross[j, i]
                xi = []
                yi = []
                zi = []
                for t in range(0, dayCount):  # 'Box Cut', removes all values more than 9 degrees in x or y.
                    idx = (abs(yiX[t] - y) <= MaxRadius)
                    xi.append(np.radians(xiX[t][idx]-x))
                    yi.append(np.radians(yiX[t][idx]-y))
                    zi.append(ziX[t][idx])

                TOT, E, V = time_loop(xi, yi, zi, np.radians(a0), np.radians(a1), dayCount)
                # print(time.time()-S2)
                mm = np.linalg.solve(E, np.asmatrix(TOT).T)

                af = mm[0]/(mm[1]+mm[0])
                mf = mm[1]+mm[0]

                STD[j, i] = mf
                LSR[j, i] = af
                obs[j, i] = V - mf
        LSR[LSR == 0] = np.nan
        STD[STD == 0] = np.nan
        obs[obs == 0] = np.nan

        print(time.time()-S1)
        TIME.append(time.time()-S1)
        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s\%s' % (Typ.upper(), Season))
        if os.path.isdir('%i/Data' % (100/N)) is False:
            os.makedirs('%i/Data' % (100/N))
        np.save(r"%i\Data\IPA_Sdv_Real.npy" % (100/N), STD)
        np.save(r"%i\Data\IPA_Obs_Real.npy" % (100/N), obs)
        np.save(r"%i\Data\IPA_Lsr_Real.npy" % (100/N), LSR)

os.chdir('../')
txt = open('IPA Diagnostics-Real.txt', 'w+')
txt.write('    Time    |    N     |    Season    |    Obs    \n')
m = 0
for N in nlist:
    for p2 in range(len(Seasonlist)):
        txt.write('  %7.2f   |  %5.2f   |     %s    |    %i\n' % (
            TIME[p2+m], 1/N, Seasonlist[p2], int(size[p2+m])))
    m += len(Seasonlist)
txt.close()
