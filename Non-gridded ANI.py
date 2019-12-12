import numpy as np
import os
import math
import datetime
import numpy.ma as ma
import time
import matplotlib.pyplot as plt
from numba import jit, njit


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


# @jit('Tuple((f8, f8[:,:], f8))(f8[:], f8[:], f4[:], f8, f8, f4, i8, i8)', forceobj=True)
def time_loop(xti, yti, zti, xx, yy, i0, i1, daycount):
    tot = np.zeros(4)
    v, vn = 0.0, 0.0
    e = np.zeros((4, 4))
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
        xxi = (degrees(2 * arcsin(sqrt(cos(radians(yyi))*cos(radians(yy)) * sin((radians(xxi - xx) / 2))**2))))**2
        yyi = (yyi - yy)**2  # Positive and negative, not absolute because squared later.
        v0 = zzi[rmin]
        tot, e, yaa, ybb = inner(xxi, yyi, zzi, v0, tot, e, i0, i1)
    return tot, e


@njit('Tuple((f8[:], f8[:,:], f8[:], f8[:]))(f8[:], f8[:], f4[:], f4, f8[:], f8[:,:], f8, i8)', parallel=True, fastmath=True)
def inner(xxi, yyi, zzi, v0, tot, e, i0, i1):

    exp = np.exp

    idx = (xxi + yyi) < 81

    xxi = xxi[idx]
    yyi = yyi[idx]
    z = zzi[idx]

    idx = (z != v0)

    xxi = xxi[idx]
    yyi = yyi[idx]
    z = z[idx]

    yaa = exp(-xxi / (2 * i0 ** 2)) * exp(-yyi / (2 * i0 ** 2))
    yab = exp(-xxi / (2 * i0 ** 2)) * exp(-yyi / (2 * i1 ** 2))
    yba = exp(-xxi / (2 * i1 ** 2)) * exp(-yyi / (2 * i0 ** 2))
    ybb = exp(-xxi / (2 * i1 ** 2)) * exp(-yyi / (2 * i1 ** 2))

    ia0 = sum(yaa * z * v0)
    ia1 = sum(ybb * z * v0)
    ia2 = sum(yab * z * v0)
    ia3 = sum(yba * z * v0)

    e[0, 0] += sum(yaa * yaa)
    e[1, 0] += sum(yaa * ybb)
    e[0, 1] += sum(yaa * ybb)
    e[2, 0] += sum(yaa * yab)
    e[0, 2] += sum(yaa * yab)
    e[3, 0] += sum(yaa * yba)
    e[0, 3] += sum(yaa * yba)

    e[1, 1] += sum(ybb * ybb)
    e[1, 2] += sum(ybb * yab)
    e[2, 1] += sum(ybb * yab)
    e[1, 3] += sum(ybb * yba)
    e[3, 1] += sum(ybb * yba)

    e[2, 2] += sum(yab * yab)
    e[2, 3] += sum(yab * yba)
    e[3, 2] += sum(yab * yba)

    e[3, 3] += sum(yba * yba)

    tot[0] += ia0
    tot[1] += ia1
    tot[2] += ia2
    tot[3] += ia3

    return tot, e, yaa, ybb


#  Creating the coarse grid for storing the innovations and producing output. In both 1d and 2d arrays.

stepy = int((32 - 8) / 0.274)
yedges = np.linspace(8, 32, int(stepy))
stepx = int((75 - 45) / 0.3)
xedges = np.linspace(45, 75, int(stepx))
ycenter = 0.5 * (yedges[1:] + yedges[:-1])
xcenter = 0.5 * (xedges[1:] + xedges[:-1])

# Gaussian projection length-scale
a0 = 0.25
a1 = 4

Overwrite = True

os.chdir(r'\\POFCDisk1\PhD_Lewis\H-L_Variances\Innovations\coarse_grid_sst\2014')

# Time variables for diagnostics
dt = datetime.timedelta(hours=24)
TIME = np.zeros(4)
tim = 0
size = np.zeros(4)
for Season in ["1-DJF", "2-MAM", "3-JJA", "4-SON"]:
    Zi = []
    Yi = []
    Xi = []
    S1 = time.time()

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

    if Overwrite is True:
        for t in range(dayCount):
            date = Start + t * dt
            # Open and save all innovations,

            dat = np.load('%i/%04i-%02i-%02i.npz' % (date.year, date.year, date.month, date.day))
            size[tim] += + len(dat['zi'])
            Zi.append(dat['zi'][:, 0])
            Xi.append(dat['xi'])
            Yi.append(dat['yi'])
            dat.close()

    os.chdir('../%s/' % Season)

    # Load coarse grid for domain and mask
    GRID = np.load('coarse_grid_sst.npz')
    gridz = GRID['zi']
    gridz = ma.masked_invalid(gridz)
    GRID.close()

    ross = np.load('rossby.npy')
    ross = ma.masked_array((ross / 1000) / (40000 / 360), gridz.mask)

    M0, M1, M2, M3 = np.zeros_like(gridz), np.zeros_like(gridz), np.zeros_like(gridz), np.zeros_like(gridz)

    if Overwrite is True:
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

                T, E = time_loop(Xi, Yi, Zi, x, y, a0, a1, dayCount)

                M = np.asmatrix(E).I

                mm = M * np.asmatrix(T).T
                M0[j, i] = mm[0]
                M1[j, i] = mm[1]
                M2[j, i] = mm[2]
                M3[j, i] = mm[3]

                print(time.time() - S2)

        os.chdir('//POFCDisk1/PhD_Lewis/EEDiagnostics/%s' % Season)
        np.save("M0.npy", np.array(M0))
        np.save("M1.npy", np.array(M1))
        np.save("M2.npy", np.array(M2))
        np.save("M3.npy", np.array(M3))

    else:
        os.chdir('//POFCDisk1/PhD_Lewis/EEDiagnostics/%s' % Season)
        M0 = np.load("M0.npy")
        M1 = np.load("M1.npy")
        M2 = np.load("M2.npy")
        M3 = np.load("M3.npy")

    V = M0 + M1 + M2 + M3
    W1 = (M0 + M1)/V
    V1 = M0/(M0 + M1)
    V2 = M2/(M2 + M3)
    plt.figure(1)
    plt.pcolormesh(V, cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(V[~np.isnan(V)], 5), np.percentile(V[~np.isnan(V)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('V.png')
    plt.close()

    plt.figure(2)
    plt.pcolormesh(W1, cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(W1[~np.isnan(W1)], 5), np.percentile(W1[~np.isnan(W1)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('W1.png')
    plt.close()

    plt.figure(3)
    plt.pcolormesh(V1, cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(V1[~np.isnan(V1)], 5), np.percentile(V1[~np.isnan(V1)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('V1.png')
    plt.close()

    plt.figure(4)
    plt.pcolormesh(V2, cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(V2[~np.isnan(M3)], 5), np.percentile(V2[~np.isnan(V2)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('V2.png')
    plt.close()

    TIME[tim] = time.time() - S1
    tim += 1
if Overwrite is True:
    os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics')
    txt = open('ANI - Analysis.txt', 'w+')
    txt.write('    Time    |    a0    |    a1    |    Season    |    Obs    \n')
    for p2 in range(4):
        txt.write('  %7.2f   |  %5.2f   |  %5.2f   |     %s    |    %i\n' % (
            TIME[p2], a0, a1, ["1-DJF", "2-MAM", "3-JJA", "4-SON"][p2], size[p2]))
    txt.close()
