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


@njit('(f8[:])(f8[:], f8[:], f8, f8)', parallel=True, fastmath=True)
def distance(la1, lo1, lat2, lon2):
    # Computes the Haversine distance between two points.

    radians, sin, cos, arcsin, sqrt, degrees = np.radians, np.sin, np.cos, np.arcsin, np.sqrt, np.degrees

    x0 = radians(lo1)
    y0 = radians(la1)
    xr = radians(lon2)
    yr = radians(lat2)

    a = sin((yr - y0)/2.0)**2.0 + (cos(y0)*cos(yr)*(sin((xr - x0)/2.0)**2.0))
    angle2 = 2.0*arcsin(sqrt(a))
    angle2 = degrees(angle2)

    return angle2


@njit('Tuple((f8, f8, f8[:,:]))(f8[:], f8[:], f4[:], f4, f4, i8, f8[:,:], f8, f8)', parallel=True, fastmath=True)
def inner(xxi, yyi, zzi, v0, i0, i1, e, t0, t1):

    idd = (yyi + xxi) < 81
    exp, sm = np.exp, np.sum
    z = zzi[idd]
    xxi = xxi[idd]
    yyi = yyi[idd]

    idd = (z != v0)

    ya = exp(-(xxi[idd] + yyi[idd]) / (2 * i0 * i0))
    yb = exp(-(xxi[idd] + yyi[idd]) / (2 * i1 * i1))

    e[0, 0] += sm(ya * ya)
    e[1, 0] += sm(ya * yb)
    e[0, 1] += sm(ya * yb)
    e[1, 1] += sm(yb * yb)

    t0 = t0 + sm(ya * z[idd] * v0)
    t1 = t1 + sm(yb * z[idd] * v0)

    return t0, t1, e


@jit('Tuple((f8, f8, f8[:,:], f8))(f8[:], f8[:], f8[:], f8, f8, f4, i8, i8)', forceobj=True)
def time_loop(xti, yti, zti, xx, yy, i0, i1, daycount):
    t0 = 0.0
    t1 = 0.0
    v, vn = 0.0, 0.0
    e = np.array([[0.0, 0.0], [0.0, 0.0]])
    degrees, sqrt, cos, sin, radians, arcsin, argmin = \
        np.degrees, np.sqrt, np.cos, np.sin, np.radians, np.arcsin, np.argmin
    for tt in range(0, daycount):
        # Set array of 'today's' innovations
        xxi = xti[tt]
        yyi = yti[tt]
        zzi = zti[tt]
        if len(zzi) <= 10:
            continue
        rmin = argmin(distance(yyi, xxi, yy, xx))
        xxi = degrees(2 * arcsin(sqrt(cos(radians(yyi))*cos(radians(yy)) * sin((radians(xxi - xx) / 2))**2)))
        yyi = yyi - yy  # Positive and negative, not absolute because squared later.
        v0 = zzi[rmin]
        t0, t1, e = inner(xxi*xxi, yyi*yyi, zzi, v0, i0, i1, e, t0, t1)
        v += v0**2
        vn += 1
    return t0, t1, e, v/vn


#  Creating the coarse grid for storing the innovations and producing output. In both 1d and 2d arrays.

stepy = int((32 - 8) / 0.274)
yedges = np.linspace(8, 32, int(stepy))
stepx = int((75 - 45) / 0.3)
xedges = np.linspace(45, 75, int(stepx))
ycenter = 0.5 * (yedges[1:] + yedges[:-1])
xcenter = 0.5 * (xedges[1:] + xedges[:-1])

Seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]
Overwrite = True
a1 = 4  # Long length-scale, in degrees.
Typ = 'sla'

#  Ross = (Ross/1000)/(40000/360) # m to degrees, according to https://stackoverflow.com/questions/5217348/how-do-i-conv
#  ert-kilometres-to-degrees-in-geodjango-geos

size = []
TIME = []
dt = datetime.timedelta(hours=24)
nlist = [100/100]  # , 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]
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
        ross = ma.masked_array((ross/1000)/(40000/360), gridz.mask)

        STD = np.array(np.zeros_like(gridz))  # Background standard Deviation
        LSR = np.array(np.zeros_like(gridz))  # Length-scale ratio
        obs = np.array(np.zeros_like(gridz))  # Observation standard deviation

        for i, x in enumerate(xcenter):
            for j, y in enumerate(ycenter):
                if gridz.mask[j, i] == True:
                    continue
                S2 = time.time()

                a0 = ross[j, i]
                xi = []
                yi = []
                zi = []
                for t in range(0, dayCount):  # 'Box Cut', removes all values more than 9 degrees in x or y.
                    idi = np.logical_and(abs(Xi[t] - x) <= 9, abs(Yi[t] - y) <= 9)
                    xi.append(Xi[t][idi])
                    yi.append(Yi[t][idi])
                    zi.append(Zi[t][idi])

                T0, T1, E, V = time_loop(xi, yi, zi, x, y, a0, a1, dayCount)
                E = np.linalg.inv(E)
                m0 = T0*(E[0, 0])+T1*(E[1, 0])
                m1 = T1*(E[1, 1])+T0*(E[0, 1])
                #
                af = m0/(m1+m0)
                mf = m1+m0

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
