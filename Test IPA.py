import numpy as np
import os
import datetime
import numpy.ma as ma
import time
import random
import matplotlib.pyplot as plt
from numba import jit, njit


@njit('(f8[:])(f8[:], f8[:], f8, f8)', parallel=True, fastmath=True)
def distance(la1, lo1, lat2, lon2):
    # Computes the Haversine distance between two points.

    radians, sin, cos, arcsin, sqrt, degrees = np.radians,np.sin, np.cos, np.arcsin, np.sqrt, np.degrees

    x1 = radians(lo1)
    y1 = radians(la1)
    x2 = radians(lon2)
    y2 = radians(lat2)

    a = sin((y2 - y1)/2.0)**2.0 + (cos(y1)*cos(y2)*(sin((x2 - x1)/2.0)**2.0))
    angle2 = 2.0*arcsin(sqrt(a))
    angle2 = degrees(angle2)

    return angle2


@njit('Tuple((f8, f8, f8[:,:]))( f8[:], f4[:], f8[:], f8[:,:], f8, f8)', parallel=True, fastmath=True)
def inner(ri, zzi, r1, z, n, m):

    idd = np.logical_and(ri > n, ri < m)

    z = np.append(z, zzi[idd])
    r1 = np.append(r1, ri[idd])
    mz = np.mean(zzi[idd])
    mr = np.mean(ri[idd])
    plt.plot(np.mean(ri[idd]), np.mean(zzi[idd]), c='k', marker='+')

    return r1, z, mz, mr


def f(c, std, lsr, i0, i1):
    return std*(lsr*np.exp(-(c*c)/(2*i0*i0)) + (1-lsr)*np.exp(-(c*c)/(2*i1*i1)))


@jit('Tuple((f8, f8, f8[:,:], f8))(f8[:], f8[:], f8[:], f8, f8, f4, i8, i8, f8, f8)', forceobj=True)
def time_loop(xti, yti, zti, xx, yy, i0, i1, daycount, std, lsr):
    r1 = np.array(0)
    z = np.array(0)
    mr = np.array(0)
    mz = np.array(0)
    n = 0
    m = 4

    degrees, sqrt, cos, sin, radians, arcsin, argmin = \
        np.degrees, np.sqrt, np.cos, np.sin, np.radians, np.arcsin, np.argmin
    for tt in range(0, daycount):
        # Set array of 'today's' innovations
        xxi = xti[tt]
        yyi = yti[tt]
        zzi = zti[tt]

        r = distance(yyi, xxi, yy, xx)
        rmin = argmin(r)

        v0 = zzi[rmin]
        r1, z, mzt, mrt = inner(r, zzi*v0, r1, z, n, m)
        mz = np.append(mz, mzt)
        mr = np.append(mr, mrt)

    r = np.linspace(n, m, 100)

    plt.plot(r, f(r, std, lsr, i0, i1))
    plt.scatter(r1[1:], z[1:], c='r', s=0.1)
    plt.figure(2)
    plt.hist(mz[1:], bins=20)
    plt.show()
    return


#  Creating the coarse grid for storing the innovations and producing output. In both 1d and 2d arrays.
stepy = int((32 - 8) / 0.274)
yedges = np.linspace(8, 32, int(stepy))
stepx = int((75 - 45) / 0.3)
xedges = np.linspace(45, 75, int(stepx))
ycenter = 0.5 * (yedges[1:] + yedges[:-1])
xcenter = 0.5 * (xedges[1:] + xedges[:-1])

x2 = np.zeros((len(xedges) - 1, len(yedges) - 1))
for p in range(0, len(yedges) - 1, 1):
    x2[:, p] = xedges[:-1]
x2 = np.transpose(x2)

y2 = np.zeros((len(yedges) - 1, len(xedges) - 1))
for p in range(0, len(xedges) - 1, 1):
    y2[:, p] = yedges[:-1]

Seasonlist = ["1-DJF"]  # , "2-MAM", "3-JJA", "4-SON"]
Overwrite = True
a1 = 4  # Long length-scale, in degrees.

#  Ross = (Ross/1000)/(40000/360) # m to degrees, according to https://stackoverflow.com/questions/5217348/how-do-i-conv
#  ert-kilometres-to-degrees-in-geodjango-geos

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
        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\coarse_grid_sst')

        random.seed(100)

        if Overwrite is True:
            for t in range(dayCount):
                date = Start + t * dt
                # Open and save all innovations,
                dat = np.load('%i/%04i-%02i-%02i.npz' % (date.year, date.year, date.month, date.day))
                idx = sorted(random.sample(list(np.arange(len(dat['zi'][:, 0]))), k=int(len(dat['zi'][:, 0]) / N)))
                Zi.append(dat['zi'][idx, 0])
                Xi.append(dat['xi'][idx])
                Yi.append(dat['yi'][idx])
        os.chdir('../%s/' % Season)

        # Load coarse grid for domain and mask
        GRID = np.load('coarse_grid_sst.npz')
        gridz = GRID['zi']
        gridz = ma.masked_invalid(gridz)
        GRID.close()

        #  Load rossby radius in terms of the coarse grid. Commented sections kept incase of need for recalculation.
        # ross = interpolate.griddata((rlon, rlat), np.array(Ross), (x2, y2), 'nearest')

        ross = np.load('rossby.npy')
        ross = ma.masked_array((ross/1000)/(40000/360), gridz.mask)
        os.chdir('../../%s/%i/Data' % (Season, 100/N))
        STD = np.load('IPA_Sdv_Real.npy')  # Background standard Deviation
        LSR = np.load('IPA_Lsr_Real.npy')  # Length-scale ratio
        obs = np.load('IPA_Obs_Real.npy')  # Observation standard deviation

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
                    idd = np.logical_and(abs(Xi[t] - x) <= 9, abs(Yi[t] - y) <= 9)
                    xi.append(Xi[t][idd])
                    yi.append(Yi[t][idd])
                    zi.append(Zi[t][idd])

                time_loop(xi, yi, zi, x, y, a0, a1, dayCount, STD[j, i], LSR[j, i])
                print(1)
        LSR[LSR == 0] = np.nan
        STD[STD == 0] = np.nan
        obs[obs == 0] = np.nan

        print(time.time()-S1)
        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s' % Season)
        if os.path.isdir('%i/test' % (100/N)) is False:
            os.makedirs('%i/test' % (100/N))
        np.save(r"%i/test/IPA_cgrid_sdv.npy" % (100/N), STD)
        np.save(r"%i/test/IPA_cgrid_lsr.npy" % (100/N), LSR)
        np.save(r"%i/test/IPA_cgrid_obs.npy" % (100/N), obs)
