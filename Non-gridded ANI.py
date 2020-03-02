import numpy as np
import os
import datetime
import numpy.ma as ma
import time
import matplotlib.pyplot as plt
from numba import jit

KM2D = 40000/360
MaxRadius = 900/KM2D


@jit('Tuple((f8, f8[:,:]))(f8[:], f8[:], f4[:], f8, f8, i8)',
     nopython=True, parallel=True, forceobj=True)
def time_loop(xti, yti, zti, i0, i1, daycount):
    tot = np.zeros(4)
    e = np.zeros((4, 4))
    for tt in range(0, daycount):
        # Set array of 'today's' innovations

        xxi = xti[tt]
        yyi = yti[tt]
        zzi = zti[tt]
        if len(zzi) <= 1:
            continue

        xx2 = xxi*xxi
        yy2 = yyi*yyi

        v0idx = np.argmin(xx2 + yy2)
        tot, e = inner(xx2, yy2, zzi, v0idx, tot, e, i0, i1)
    return tot, e


@jit('Tuple((f8[:], f8[:,:]))(f8[:], f8[:], f4[:], f8, f8[:], f8[:,:], f8, f8)',
     nopython=True, parallel=True, fastmath=True)
def inner(xx2, yy2, zzi, v0idx, tot, e, i0, i1):

    exp, summ = np.exp, np.sum

    idd = (xx2 + yy2) < MaxRadius*MaxRadius  # First filter of points that are, at least, MaxRadius away from the centre
    idd[v0idx] = False  # Removes the element that corresponds to v0 because the observation correlation is not zero.

    xx2 = xx2[idd]
    yy2 = yy2[idd]
    z   = zzi[idd]

    il0 = 1.0/(2*i0*i0)
    il1 = 1.0/(2*i1*i1)

    expx0 = exp(-xx2*il0)
    expy0 = exp(-yy2*il0)
    expx1 = exp(-xx2*il1)
    expy1 = exp(-yy2*il1)
    yaa = expx0*expy0
    yab = expx0*expy1
    yba = expx1*expy0
    ybb = expx1*expy1

    v0 = zzi[v0idx]
    tot[0] += v0*summ(yaa * z)
    tot[1] += v0*summ(ybb * z)
    tot[2] += v0*summ(yab * z)
    tot[3] += v0*summ(yba * z)

    e00 = summ(yaa*yaa)
    e10 = summ(yaa*ybb)
    e20 = summ(yaa*yab)
    e30 = summ(yaa*yba)
    e[0, 0] += e00
    e[1, 0] += e10
    e[0, 1] += e10
    e[2, 0] += e20
    e[0, 2] += e20
    e[3, 0] += e30
    e[0, 3] += e30

    e11 = summ(ybb*ybb)
    e12 = summ(ybb*yab)
    e13 = summ(ybb*yba)
    e[1, 1] += e11
    e[1, 2] += e12
    e[2, 1] += e12
    e[1, 3] += e13
    e[3, 1] += e13

    e22 = summ(yab*yab)
    e23 = summ(yab*yba)
    e[2, 2] += e22
    e[2, 3] += e23
    e[3, 2] += e23

    e33 = summ(yba*yba)
    e[3, 3] += e33

    return tot, e


#  Creating the coarse grid for storing the innovations and producing output. In both 1d and 2d arrays.

stepy = int((32 - 8) / 0.274)
yedges = np.linspace(8, 32, int(stepy))
stepx = int((75 - 45) / 0.3)
xedges = np.linspace(45, 75, int(stepx))
ycenter = 0.5 * (yedges[1:] + yedges[:-1])
xcenter = 0.5 * (xedges[1:] + xedges[:-1])

# Gaussian projection length-scale
a1 = 4.0
Typ = 'sst'
Seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]
Overwrite = True

# Time variables for diagnostics
dt = datetime.timedelta(hours=24)
TIME = np.zeros(4)
tim = 0
size = np.zeros(4)
for Season in Seasonlist:
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
    os.chdir(r'\\POFCDisk1\PhD_Lewis\ErrorEstimation\Preprocessed\Innovations_%s' % Typ)

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
    GRID = np.load('coarse_grid_%s.npz' % Typ)
    gridz = GRID['zi']
    gridz = ma.masked_invalid(gridz)
    GRID.close()

    ross = np.load('rossby.npy')
    ross = ma.masked_array((ross / 1000) / KM2D, gridz.mask)

    M0, M1, M2, M3 = np.zeros_like(gridz), np.zeros_like(gridz), np.zeros_like(gridz), np.zeros_like(gridz)

    if Overwrite is True:
        for i, y in enumerate(ycenter):

            print(i, time.time() - S1)

            xiY = []
            yiY = []
            ziY = []
            # First filters the innovations in the range x-MaxRadius..x+MaxRadius
            for t in range(dayCount):
                idx = (abs(Yi[t] - y) <= MaxRadius)
                xiY.append(Xi[t][idx])
                yiY.append(Yi[t][idx])
                ziY.append(Zi[t][idx])

            for j, x in enumerate(xcenter):
                if gridz.mask[i, j] == True:
                    continue

                # Secondly filters the innovations in the range y-MaxRadius..y+MaxRadius
                xi = []
                yi = []
                zi = []
                for t in range(dayCount):
                    idx = abs(xiY[t] - x) <= MaxRadius
                    xi.append(xiY[t][idx] - x)
                    yi.append(yiY[t][idx] - y)
                    zi.append(ziY[t][idx])

                S2 = time.time()

                a0 = ross[i, j]

                T, E = time_loop(xi, yi, zi, a0, a1, dayCount)

                mm = np.linalg.solve(E, np.asmatrix(T).T)

                M0[i, j] = mm[0]
                M1[i, j] = mm[1]
                M2[i, j] = mm[2]
                M3[i, j] = mm[3]

        os.chdir('//POFCDisk1/PhD_Lewis/ErrorEstimation/%s/%s/ANI' % (Typ.upper(), Season))
        np.save("M0.npy", np.array(M0))
        np.save("M1.npy", np.array(M1))
        np.save("M2.npy", np.array(M2))
        np.save("M3.npy", np.array(M3))
        TIME[tim] = time.time() - S1
        tim += 1

for Season in Seasonlist:
    os.chdir('//POFCDisk1/PhD_Lewis/ErrorEstimation/%s/%s/ANI' % (Typ.upper(), Season))
    M0 = np.load("M0.npy")
    M1 = np.load("M1.npy")
    M2 = np.load("M2.npy")
    M3 = np.load("M3.npy")

    for M in [M0, M1, M2, M3]:
        M[M == 0] = np.nan

    V = M0 + M1 + M2 + M3
    W1 = (M0 + M1)/V
    V1 = M0/(M0 + M1)
    V2 = M2/(M2 + M3)
    plt.figure(1)
    plt.pcolormesh(np.sqrt(V), cmap='jet', vmin=0.2, vmax=0.8)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(np.sqrt(V[~np.isnan(np.sqrt(V))]), 5),
                                              np.percentile(np.sqrt(V[~np.isnan(np.sqrt(V))]), 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('V.png')
    plt.close()

    plt.figure(2)
    plt.pcolormesh(W1, cmap='jet', vmin=0, vmax=1.0)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(W1[~np.isnan(W1)], 5),
                                              np.percentile(W1[~np.isnan(W1)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('W1.png')
    plt.close()

    plt.figure(3)
    plt.pcolormesh(V1, cmap='jet', vmin=0, vmax=1.0)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(V1[~np.isnan(V1)], 5),
                                           np.percentile(V1[~np.isnan(V1)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('V1.png')
    plt.close()

    plt.figure(4)
    plt.pcolormesh(V2, cmap='jet', vmin=-2, vmax=2)
    plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(V2[~np.isnan(V2)], 5),
                                              np.percentile(V2[~np.isnan(V2)], 95)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('V2.png')
    plt.close()

if Overwrite is True:
    os.chdir(r'\\POFCDisk1\PhD_Lewis\ErrorEstimation\%s' % Typ.upper())
    txt = open('ANI - Analysis.txt', 'w+')
    txt.write('    Time    |    a0    |    a1    |    Season    |    Obs    \n')
    for p2 in range(4):
        txt.write('  %7.2f   |  %5.2f   |  %5.2f   |     %s    |    %i\n' % (
            TIME[p2], 0.25, a1, ["1-DJF", "2-MAM", "3-JJA", "4-SON"][p2], size[p2]))
    txt.close()
