import time
import os
import random
import numpy as np
import numpy.ma as ma
import datetime
from numba import jit, njit

#   HL-Process_Innovations.py is a python script created by Lewis Sampson. 3 of 6, Third.
#
#   The purpose of this script is calculate the error correlation value at zero separation for each grid point,
#   according to the H-L method of error estimation.
#
#   This method; is limited by a distance, has an option to overwrite current data for specific season, and writes out
#   a .txt file with basic diagnostic information.
#
#   It is required that the coarse_grid.npz be up-to-date, for the chosen season, as this will determine which
#   grid points the correlation is calculated for. This script also needs all available coarse_grid_sst dates to be
#   processed


@njit('f8[:,:](f8, f8, f8[:], f8[:])', parallel=True, fastmath=True)
def distance(la1, lo1, lat2, lon2):
    # Computes the Haversine distance between two points.

    lon1 = np.zeros((len(lon2) , len(lat2)))
    for p in range(0, len(lat2) , 1):
        lon1[:, p] = lon2[:]
    lon1 = np.transpose(lon1)

    lat1 = np.zeros((len(lat2) , len(lon2)))
    for p in range(0, len(lon2) , 1):
        lat1[:, p] = lat2[:]

    radians, sin, cos, arcsin, sqrt, degrees = np.radians,np.sin, np.cos, np.arcsin, np.sqrt, np.degrees

    x1 = radians(lo1)
    y1 = radians(la1)
    x2 = radians(lon1)
    y2 = radians(lat1)

    a = sin((y2 - y1)/2.0)**2.0 + (cos(y1)*cos(y2)*(sin((x2 - x1)/2.0)**2.0))
    angle2 = 2.0*arcsin(sqrt(a))
    angle2 = degrees(angle2)

    return angle2


@jit('f8[:,:,:](i8, i8[:], i8[:], i8[:], f8[:], f8[:,:])', forceobj=True)
def griddata(daycount, dayidx, latidx, lonidx, sst, gridmean):
    # Combines all the data into a 2D grid that has the size of the computational grid times the number of days.
    # The values at each entry is the average of all the values (minus the global mean) in one day that fall into each grid cell.

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


@njit('Tuple((f8, f8, f8[:,:]))(f8[:], f8[:], f8, f4, i8, f8[:,:], f8, f8)', parallel=True, fastmath=True)
def inner(d, z, v0, i0, i1, e, t0, t1):
    exp, sm = np.exp, np.sum

    idd = (z != v0)

    ya = exp(-(d[idd] * d[idd]) / (2 * i0 * i0))
    yb = exp(-(d[idd] * d[idd]) / (2 * i1 * i1))

    e[0, 0] += sm(ya * ya)
    e[1, 0] += sm(ya * yb)
    e[0, 1] += sm(ya * yb)
    e[1, 1] += sm(yb * yb)

    t0 = t0 + sm(ya * z[idd] * v0)
    t1 = t1 + sm(yb * z[idd] * v0)
    t0 = t0 + sm(ya * z[idd] * v0)
    t1 = t1 + sm(yb * z[idd] * v0)
    return t0, t1, e


@jit('Tuple((f8, f8, f8[:,:], f8))(f8[:], f8[:], f8[:,:,:], f8, f8, f4, i8, i8)', forceobj=True)
def time_loop(xti, yti, zti, xx, yy, i0, i1, daycount):
    t0 = 0.0
    t1 = 0.0
    v, vn = 0.0, 0.0
    e = np.array([[0.0, 0.0], [0.0, 0.0]])
    for tt in range(0, daycount):
        # Set array of 'today's' innovations

        zzi = zti[:,:,tt]
        d = distance(yti[yy], xti[xx], yti, xti)
        v0 = zzi[yy, xx]
        if ~np.isnan(v0):
            idd = np.where(~np.isnan(zzi))
            z = zzi[idd]
            d = d[idd]
            idd = np.sqrt(d * d) < 9
            z = z[idd]
            d = d[idd]
            t0, t1, e = inner(d, z, v0, i0, i1, e, t0, t1)
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
a1 = 4

Seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]
nlist = [100/100, 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]
dt = datetime.timedelta(hours=24) #  Time Step.
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
        S1 = time.time() #  Time diagnostics

        os.chdir('%s' % season)
        GRID  = np.load('coarse_grid_sst.npz')
        Bias  = GRID['zi']
        gridz = ma.masked_invalid(Bias)
        gridindices = np.where(gridz.mask == False)
        GRID.close()

        #  Load rossby radius in terms of the coarse grid. Commented sections kept incase of need for recalculation.
        # ross = interpolate.griddata((rlon, rlat), np.array(Ross), (x2, y2), 'nearest')

        ross = np.load('rossby.npy')
        ross = ma.masked_array((ross/1000)/(40000/360), gridz.mask)
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
        obs =  np.zeros(Bias.shape)
        LSR = np.zeros(Bias.shape)

        for j, lo2 in enumerate(xcenter):
            for i, la2 in enumerate(ycenter):

                if overwrite is False:
                    continue
                if gridz.mask[i,j] == True:
                    continue
                S2 = time.time()
                a0 = ross[i, j]
                try:
                    T0, T1, E, V = time_loop(xcenter, ycenter, allGriddedValues, j, i, a0, a1, dayCount)
                    E = np.linalg.inv(E)
                    m0 = T0 * (E[0, 0]) + T1 * (E[1, 0])
                    m1 = T1 * (E[1, 1]) + T0 * (E[0, 1])

                    af = m0 / (m1 + m0)
                    mf = m1 + m0

                    STD[i, j] = mf
                    LSR[i, j] = af
                    obs[i, j] = V - mf
                except:
                    STD[i, j] = 0
                    LSR[i, j] = 0
                    obs[i, j] = 0

        STD[STD == 0] = np.nan
        LSR[LSR == 0] = np.nan
        obs[obs == 0] = np.nan
        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s' % season)
        if os.path.isdir('%i' % (100/N)) is False:
            os.makedirs('%i' % (100/N))
        np.save(r'%i\Data\IPA_Sdv_Grid.npy' % (100/N), STD)
        np.save(r'%i\Data\IPA_Lsr_Grid.npy' % (100/N), LSR)
        np.save(r'%i\Data\IPA_Obs_Grid.npy' % (100/N), obs)
        print(time.time()-S1)
        size.append(Count)
        TIME.append(time.time()-S1)

os.chdir('../')
txt = open('IPA Diagnostics-Gridded.txt', 'w+')
txt.write('    Time    |    N     |    Season    |    Obs    \n')
m = 0
for N in nlist:
    for p2 in range(len(Seasonlist)):
        txt.write('  %7.2f   |  %5.2f   |     %s    |    %i\n' % (
            TIME[p2 + m], 1/N, Seasonlist[p2], size[p2 + m]))
    m += len(Seasonlist)
txt.close()


