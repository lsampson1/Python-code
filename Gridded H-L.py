import time
import os
import random
import numpy as np
import numpy.ma as ma
import datetime
from scipy.optimize import curve_fit
from numba import jit, njit

########################################################################################################################
#
#  Gridded H-L.npy
#
#  This script runs the gridded 1D isotropic Hollingsworth and Lonnberg for Seasons in
#  Seasonlist, with N percentage of observations. Prior to this script running there are
#  files created in \\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\ and these are
#  required for each season run. This includes .npz files for observations,
#  coarse_grid_sst.npz (for the chosen seasons), and a rossby radius .npy file
#
########################################################################################################################


@njit('f8[:,:](f8, f8, f8[:], f8[:])', parallel=True, fastmath=True)
def distance(la1, lo1, lat2, lon2):
    # Computes the Haversine distance between two points.

    lon1 = np.zeros((len(lon2), len(lat2)))
    for pr in range(0, len(lat2), 1):
        lon1[:, pr] = lon2[:]
    lon1 = np.transpose(lon1)

    lat1 = np.zeros((len(lat2), len(lon2)))
    for pr in range(0, len(lon2), 1):
        lat1[:, pr] = lat2[:]

    radians, sin, cos, arcsin, sqrt, degrees = np.radians, np.sin, np.cos, np.arcsin, np.sqrt, np.degrees

    x0 = radians(lo1)
    y0 = radians(la1)
    xr = radians(lon1)
    yr = radians(lat1)

    a = sin((yr - y0)/2.0)**2.0 + (cos(y0)*cos(yr)*(sin((xr - x0)/2.0)**2.0))
    angle2 = 2.0*arcsin(sqrt(a))
    angle2 = degrees(angle2)

    return angle2


@jit('f8[:,:,:](i8, i8[:], i8[:], i8[:], f8[:], f8[:,:])', forceobj=True)
def griddata(daycount, dayidx, latidx, lonidx, sst, gridmean):
    # Combines all the data into a 2D grid that has the size of the computational grid times the number of days.
    # The values at each entry is the average of all the values (minus the global mean) in one day that fall into
    # each grid cell.

    allgriddedvalues = np.zeros((gridmean.shape[0], gridmean.shape[1], daycount))

    for day in range(daycount):
        valcount = np.zeros(gridmean.shape)
        for k in range(dayidx[day], dayidx[day + 1]):
            laidx2 = latidx[k]
            loidx2 = lonidx[k]

            allgriddedvalues[laidx2, loidx2, day] += sst[k]
            valcount[laidx2, loidx2] += 1
        valcount2 = valcount.copy()
        idx = (valcount2 == 0)
        valcount2[idx] = 1.00
        allgriddedvalues[:, :, day] = allgriddedvalues[:, :, day]/valcount2
        allgriddedvalues[:, :, day][np.isnan(gridmean)] = np.nan
        allgriddedvalues[valcount == 0, day] = np.nan

    return allgriddedvalues


@jit('Tuple((f8[:],f8[:]))(f8, i8, i8, f8[:,:,:], f8[:], f8[:], f8[:,:])', forceobj=True)
def fit(dd, laidx1, loidx1, allgriddedvalues, lat, lon, fitvar):

    fitcorr = np.zeros(int(30)+1)
    fitcorrcount = np.zeros(int(30)+1)

    la1 = lat[laidx1]
    lo1 = lon[loidx1]
    if (fitvar[laidx1, loidx1] < 1e-10) or np.isnan(allgriddedvalues[laidx1, loidx1, :]).all():
        return fitcorr, fitcorr

    d = distance(la1, lo1, lat, lon)

    for laidx2, lat2 in enumerate(lat):
        for loidx2, lon2 in enumerate(lon):
            if d[laidx2, loidx2] <= 900000:
                v1 = allgriddedvalues[laidx1, loidx1, :]
                v2 = allgriddedvalues[laidx2, loidx2, :]
                if np.isnan(v1*v2).all():
                    continue
                valididx = np.logical_not(np.isnan(v1*v2))
                provcorr = np.nanmean(v1*v2)
                if not np.isnan(provcorr) and np.var(v2[valididx]) > 1e-10:
                    fitcorr[int(d[laidx2, loidx2]/dd)] += provcorr
                    fitcorrcount[int(d[laidx2, loidx2]/dd)] += 1
    fitcorrcount[fitcorrcount == 0] = np.nan
    return (fitcorr/fitcorrcount).flatten(), np.arange(31)*dd


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

#  Loading the Rossby radius of deformation, to assign the shorter length-scale.
# os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed')
# R = Dataset('rossby_radii.nc', 'r')
# ross = np.array(R.variables['rossby_r'][:])
# rlat = np.array(R.variables['lat'][:])
# rlon = np.array(R.variables['lon'][:])
# R.close()
# rlat = rlat[np.where(np.array(ross) <= 200000)]
# rlon = rlon[np.where(np.array(ross) <= 200000)]
# Ross = ross[np.where(ross <= 200000)]

overwrite = True
Dx = 30000  # Bin size in metres.

Seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]
nlist = [100/100, 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]
dt = datetime.timedelta(hours=24)  # Time Step.
TIME = []
size = []
for N in nlist:
    si = 0
    for season in Seasonlist:
        if season == '1-DJF':
            Start = datetime.date(2013, 12,  1)
            End = datetime.date(2014, 3, 1)
        elif season == '2-MAM':
            Start = datetime.date(2014, 3, 1)
            End = datetime.date(2014, 6, 1)
        elif season == '3-JJA':
            Start = datetime.date(2014, 6, 1)
            End = datetime.date(2014, 9, 1)
        elif season == '4-SON':
            Start = datetime.date(2014, 9, 1)
            End = datetime.date(2014, 12, 1)
        else:
            Start = 0
            End = 0
            exit(0)

        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed')
        S1 = time.time()  # Time diagnostics

        os.chdir('%s' % season)
        GRID = np.load('coarse_grid_sst.npz')
        Bias = GRID['zi']
        gridz = ma.masked_invalid(Bias)
        gridindices = np.where(gridz.mask == False)
        GRID.close()

        #  Load rossby radius in terms of the coarse grid. Commented sections kept incase of need for recalculation.
        # ross = interpolate.griddata((rlon, rlat), np.array(Ross), (x2, y2), 'nearest')

        ross = np.load('rossby.npy')
        ross = ma.masked_array(ross, gridz.mask)
        os.chdir('../')

        #  Gather information about the number of observations that occur on each day.

        InnovationCount = 0
        dayCount = (End - Start).days
        dayIdx = np.zeros(dayCount + 1, np.int64)
        for t in range(dayCount):
            date = Start + t * dt
            cg = np.load('coarse_grid_sst/%s/%s.npz' % (date.year, date))
            dayIdx[t] = InnovationCount
            InnovationCount += int(len(cg['xi'])/N)
        dayIdx[dayCount] = InnovationCount

        #  Prepare arrays for assignment.

        SST = np.zeros((InnovationCount, 1))
        Lat = np.zeros(InnovationCount)
        Lon = np.zeros(InnovationCount)
        dayNum = np.zeros(InnovationCount, np.int)

        ValidDays = 0
        Count = 0
        Var = np.zeros_like(gridz)

        random.seed(100)

        #  Assign array equal to the amount of observations that day, dayNum saves what day these values relate to.
        #  Observations chosen as a random 1/N % of the whole data set. Seeded with 100.

        for t in range(dayCount):
            date = Start + t * dt
            cg = np.load('coarse_grid_sst/%s/%s.npz' % (date.year, date))
            Idx = sorted(random.sample(list(np.arange(len(cg['xi']))), k=int(len(cg['xi'][:])/N)))
            xi, yi, zi = cg['xi'][Idx], cg['yi'][Idx], cg['zi'][Idx, :]
            dataSize = xi.size
            SST[Count:Count + dataSize] = zi
            Lat[Count:Count + dataSize] = yi
            Lon[Count:Count + dataSize] = xi
            dayNum[Count:Count + dataSize] = t
            Count += dataSize

        latIdx = np.digitize(Lat, yedges) - 1
        lonIdx = np.digitize(Lon, xedges) - 1

        #  Put innovations into a grid of x, y, t. This speeds up the later loops.

        allGriddedValues = griddata(dayCount, dayIdx, latIdx.flatten(), lonIdx.flatten(), SST.flatten(), Bias)

        #  Prepare arrays.

        STD = np.zeros(Bias.shape)
        obs = np.zeros(Bias.shape)
        LSR = np.zeros(Bias.shape)

        for i, la2 in enumerate(ycenter):
            for j, lo2 in enumerate(xcenter):
                if overwrite is False:
                    continue
                if gridz.mask[i,j] == True:
                    continue
                S2 = time.time()

                #  fit is the binning process, allGriddedValues is turned into a covariance array with 30 bins of 30km.
                (cov, dist) = fit(Dx, i, j, allGriddedValues, ycenter, xcenter, Bias**2)
                if (cov != 0.0).any():
                    validIdx = np.logical_not(np.isnan(cov))
                    validIdx[0] = False
                    try:
                        def func(xx, xa, xb):  # The function is created anew, as it is dependent on the Rossby radius.
                            return xa*np.exp(-(xx**2)/(2*ross[i, j]**2)) + xb*np.exp(-(xx**2)/(2*(444*1000)**2))
                        #  Apply the curve fitting, the function is from scipy.curve_fit.
                        popt, pcov = curve_fit(func, dist[validIdx], cov[validIdx], maxfev=100000000)
                        STD[i, j] = popt[0]+popt[1]
                        obs[i, j] = cov[0] - (popt[0]+popt[1])
                        LSR[i, j] = popt[0]/(popt[0]+popt[1])
                    except:
                        STD[i, j] = np.nan
                        obs[i, j] = np.nan
                        LSR[i, j] = np.nan
                else:
                    STD[i, j] = np.nan
                    obs[i, j] = np.nan
                    LSR[i, j] = np.nan
        STD[STD == 0] = np.nan
        LSR[LSR == 0] = np.nan
        obs[obs == 0] = np.nan
        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s' % season)
        if os.path.isdir('%i' % (100/N)) is False:
            os.makedirs('%i' % (100/N))
        np.save(r'%i\Data\HL_Sdv_Grid.npy' % (100/N), STD)
        np.save(r'%i\Data\HL_Lsr_Grid.npy' % (100/N), LSR)
        np.save(r'%i\Data\HL_Obs_Grid.npy' % (100/N), obs)
        print(time.time()-S1)
        size.append(Count)
        TIME.append(time.time()-S1)

os.chdir('../')
txt = open('HL Diagnostics-Gridded.txt', 'w+')
txt.write('    Time    |    N     |    Season    |    Obs    \n')
m = 0
for N in nlist:
    for p2 in range(len(Seasonlist)):
        txt.write('  %7.2f   |  %5.2f   |     %s    |    %i\n' % (
            TIME[p2 + m], 1/N, Seasonlist[p2], size[p2 + m]))
    m += len(Seasonlist)
txt.close()