import numpy as np
import os
import math
import datetime
import numpy.ma as ma
import time
import matplotlib.pyplot as plt
from numba import jit


@jit('Tuple((f8[:], f8[:,:], i8))(f8[:], f8[:], f4[:], f8, f8, f8[:], f8[:,:], i8, f8, i8)', nopython=True)
def inner(xxi, yyi, zzi, xx, yy, tot, e, tt, i0, i1):
    r0 = np.zeros_like(xxi)
    n = 0
    for u, v in zip(yyi, xxi):
        x1 = math.radians(v)
        y1 = math.radians(u)
        x2 = math.radians(xx)
        y2 = math.radians(yy)

        # Compute using the Haversine formula.
        a = math.sin((y2 - y1) / 2.0) ** 2.0 + (math.cos(y1) * math.cos(y2) * (math.sin((x2 - x1) / 2.0) ** 2.0))

        # Great circle distance in radians
        angle2 = 2.0 * math.asin(min(1.0, math.sqrt(a)))

        # Convert back to degrees.
        r0[n] = math.degrees(angle2) * (40000 / 360)
        n += 1

    ymin = yyi[np.where(r0 == np.nanmin(r0))][0]
    xmin = xxi[np.where(r0 == np.nanmin(r0))][0]

    if abs(xmin - xx) + abs(ymin - yy) >= 3:
        return tot, e, tt

    idx = r0 < 900

    lon = xxi[idx]
    lat = yyi[idx]
    z = zzi[idx]
    # r = r0[idx]/(40000/360)

    vv = sorted(z[(lon == xmin) & (lat == ymin)])
    v0 = vv[int(len(vv) / 2)]

    for k in range(len(lon)):
        lon[k] = math.degrees(2 * math.asin(min(np.sqrt(math.cos(math.radians(lat[k]))*math.cos(math.radians(yy))*math.sin((math.radians(lon[k] - xx) / 2)) ** 2), 1)))
    lat = lat - yy

    yaa = np.exp(-lon[np.where(z != v0)] ** 2 / (2 * i0 ** 2)) * np.exp(-lat[np.where(z != v0)] ** 2 / (2 * i0 ** 2))
    yab = np.exp(-lon[np.where(z != v0)] ** 2 / (2 * i0 ** 2)) * np.exp(-lat[np.where(z != v0)] ** 2 / (2 * i1 ** 2))
    yba = np.exp(-lon[np.where(z != v0)] ** 2 / (2 * i1 ** 2)) * np.exp(-lat[np.where(z != v0)] ** 2 / (2 * i0 ** 2))
    ybb = np.exp(-lon[np.where(z != v0)] ** 2 / (2 * i1 ** 2)) * np.exp(-lat[np.where(z != v0)] ** 2 / (2 * i1 ** 2))

    ia0 = np.sum(yaa * z[np.where(z != v0)]*v0)
    ia1 = np.sum(ybb * z[np.where(z != v0)]*v0)
    ia2 = np.sum(yab * z[np.where(z != v0)]*v0)
    ia3 = np.sum(yba * z[np.where(z != v0)]*v0)

    e[0, 0] += np.sum(yaa * yaa)
    e[1, 0] += np.sum(yaa * ybb)
    e[0, 1] += np.sum(yaa * ybb)
    e[2, 0] += np.sum(yaa * yab)
    e[0, 2] += np.sum(yaa * yab)
    e[3, 0] += np.sum(yaa * yba)
    e[0, 3] += np.sum(yaa * yba)

    e[1, 1] += np.sum(ybb * ybb)
    e[1, 2] += np.sum(ybb * yab)
    e[2, 1] += np.sum(ybb * yab)
    e[1, 3] += np.sum(ybb * yba)
    e[3, 1] += np.sum(ybb * yba)

    e[2, 2] += np.sum(yab * yab)
    e[2, 3] += np.sum(yab * yba)
    e[3, 2] += np.sum(yab * yba)

    e[3, 3] += np.sum(yba * yba)

    # y = np.array((np.sum(yaa), np.sum(ybb), np.sum(yab), np.sum(yba)))
    # e += np.outer(y, y)

    tot[0] += ia0
    tot[1] += ia1
    tot[2] += ia2
    tot[3] += ia3
    tt += 1

    return tot, e, tt


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
    zi = []
    yi = []
    xi = []
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
    os.chdir(r'\\POFCDisk1\PhD_Lewis\H-L_Variances\Innovations\coarse_grid_sst')

    if Overwrite is True:
        for t in range(dayCount):
            date = Start + t * dt
            # Open and save all innovations,

            dat = np.load('%i/%04i-%02i-%02i.npz' % (date.year, date.year, date.month, date.day))
            size[tim] += + len(dat['zi'])
            zi.append(dat['zi'][:, 0])
            xi.append(dat['xi'])
            yi.append(dat['yi'])
            dat.close()
            zi[zi == 0] = np.nan
            xi[xi == 0] = np.nan
            yi[yi == 0] = np.nan
            dat.close()

    os.chdir('../%s/' % Season)

    # Load coarse grid for domain and mask
    GRID = np.load('coarse_grid_sst.npz')
    gridx = GRID['xi']
    gridy = GRID['yi']
    gridz = GRID['zi']
    gridz = ma.masked_invalid(gridz)
    GRID.close()

    ycenter = 0.5 * (gridy[1:] + gridy[:-1])
    xcenter = 0.5 * (gridx[1:] + gridx[:-1])

    M0, M1, M2, M3 = np.zeros_like(gridz), np.zeros_like(gridz), np.zeros_like(gridz), np.zeros_like(gridz)

    if Overwrite is True:
        for i, x in enumerate(xcenter):
            for j, y in enumerate(ycenter):
                if gridz.mask[j, i] == True:
                    continue
                S2 = time.time()

                TT = 0
                T = np.zeros(4)
                E = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
                for t in range(1, dayCount):
                    # Set array of 'today's' innovations
                    Xi = np.array(xi[t])
                    Yi = np.array(yi[t])
                    Zi = np.array(zi[t])
                    Zi[np.where((abs(Xi - x) >= 9))] = np.nan
                    Zi[np.where((abs(Yi - y) >= 9))] = np.nan

                    Yi = Yi[np.where(np.isnan(Zi) == False)]
                    Xi = Xi[np.where(np.isnan(Zi) == False)]
                    Zi = Zi[np.where(np.isnan(Zi) == False)]

                    T, E, TT = inner(Xi, Yi, Zi, x, y, T, E, TT, a0, a1)

                if TT == 0:
                    continue

                M = E.I

                mm = np.matrix(M)*np.matrix(T).T

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

    plt.figure(1)
    plt.pcolormesh(M0, cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(M0[~np.isnan(M0)], 5), np.percentile(M0[~np.isnan(M0)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('m0.png')
    plt.close()

    plt.figure(2)
    plt.pcolormesh(M1, cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(M1[~np.isnan(M1)], 5), np.percentile(M1[~np.isnan(M1)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('m1.png')
    plt.close()

    plt.figure(3)
    plt.pcolormesh(M2, cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(M2[~np.isnan(M2)], 5), np.percentile(M2[~np.isnan(M2)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('m2.png')
    plt.close()

    plt.figure(4)
    plt.pcolormesh(M3, cmap='jet', vmin=-1, vmax=1)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(M3[~np.isnan(M3)], 5), np.percentile(M3[~np.isnan(M3)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('m3.png')
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
