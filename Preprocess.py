import time
import os
import calendar
import numpy as np
import numpy.ma as ma
from scipy import ndimage
import scipy.interpolate as interpolate
from datetime import datetime, date, timedelta
from netCDF4 import Dataset
from numba import jit
from shutil import copyfile

########################################################################################################################
# Purpose: Preprocess.py, before running the IPA and H-L scripts we store the innovations in a uniform format. We also
# produce a coarse grid for the AS20 domain, the mask for this grid is determined based on where innovations do occur.
# We also prepare the Rossby radius following this coarse grid and mask.
#
# Prerequisites:
#       - BaseDir available.
#       - YYYYMMDDT0000Z_xxx_qc_BiasCorrfb_oo_qc_fdbk.nc, for the set date and variable under the BaseDir.
#       - rossy_radii.nc, from NEMOVar suite. (Original is in ocean\OPERATIONAL_SUITE_V5.3\...)
#       - Model run output. Bmod_sdv_mxl... and Bmod_sdv_wgt... (This is to format .nc files)
#       - Unzipped assimilated model runs. e.g. diaopfoam files (This is for depth gradient). UZ.sh script will unzip.
#
# Output: Stored innovations from Start to End-1 day. Coarse masked grid and Rossby radius for Seasons.
#
########################################################################################################################

# Setup coarse grid model domain. ~30km resolution.
LatLim = [8, 32]
LonLim = [45, 75]
dlat = 0.274
dlon = 0.3

# Vectorised coarse grid.
stepy = int((LatLim[1] - LatLim[0]) / dlat)
yedges = np.linspace(LatLim[0], LatLim[1], int(stepy))
stepx = int((LonLim[1] - LonLim[0]) / dlon)
xedges = np.linspace(LonLim[0], LonLim[1], int(stepx))

# 2D matrix coarse grid.
x2 = np.zeros((len(xedges) - 1, len(yedges) - 1))
for p in range(0, len(yedges) - 1, 1):
    x2[:, p] = xedges[:-1]
x2 = np.transpose(x2)

y2 = np.zeros((len(yedges) - 1, len(xedges) - 1))
for p in range(0, len(xedges) - 1, 1):
    y2[:, p] = yedges[:-1]

print('Running Preprocess.py script.')
print('Produces all required files for error estimation scripts. Stores the innovations and coarse grid for "var" \n'
      'produces Rossby radius and TGradient for each season, and makes any necessary subdirectories. \n')
print('Time: %s ' % datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
dt = timedelta(hours=24)

@jit('(f8[:,:])(f8[:,:], i8)', forceobj=True)
def smooth_var_array(data, sigma):
    data[data == 0] = np.nan
    data = ma.masked_invalid(data)
    masks = 1 - data.mask.astype(np.float)
    wsum = ndimage.gaussian_filter(masks * data, sigma)
    gsum = ndimage.gaussian_filter(masks, sigma)

    gfilter = wsum / gsum

    gfilter = ma.masked_array(gfilter, mask=data.mask)

    return gfilter

@jit('Tuple((f8[:,:],f8[:,:],f8[:,:]))(i8, u1)', nopython=True, parallel=True, forceobj=True)
def read_observation(s, var):
    cwd = os.getcwd()
    ymd = s.strftime("%Y%m%d")
    name = '%sT0000Z_%s_qc_BiasCorrfb_oo_qc_fdbk.nc' % (ymd, var)

    data = Dataset(name, 'r')
    lat = data.variables['LATITUDE'][:]
    lon = data.variables['LONGITUDE'][:]
    obs = data.variables['SST_OBS'][:]
    mod = data.variables['SST_Hx'][:]
    obs = obs - mod

    os.chdir(cwd)
    return lat, lon, obs


@jit('(f8[:,:,:])(i8, i8[:], i8[:], i8[:], f8[:], f8[:,:])', nopython=True, parallel=True, forceobj=True)
def griddata(daycount, dayidx, latidx, lonidx, sst, gridmean):
    # Combines all the data into a 3D grid that has the size of the computational grid times the number of days.
    # The values at each entry is the average of all the values (minus the global mean) in one day that fall into
    # each grid cell.

    allgriddedvalues = np.zeros((gridmean.shape[0], gridmean.shape[1], daycount,))

    for day in range(daycount):
        valcount = np.zeros(gridmean.shape)
        for k in range(dayidx[day], dayidx[day + 1]):
            laidx2 = latidx[k]
            loidx2 = lonidx[k]

            allgriddedvalues[laidx2, loidx2, day] += sst[k]
            valcount[laidx2, loidx2] += 1
        valcount[valcount == 0] = 1
        allgriddedvalues[:, :, day] = allgriddedvalues[:, :, day] / valcount
        allgriddedvalues[:, :, day][np.isnan(gridmean)] = np.nan

    return allgriddedvalues


BaseDir = '\\\\POFCDisk1\\PhD_Lewis'
PreDir = 'EEPrerequisites'

StartList = []
EndList = []
Start = date(2014, 1, 1)
End = date(2015, 1, 1)
dayCount = (End - Start).days

StartList.append(date(2013, 12, 1))
StartList.append(date(2014, 3, 1))
StartList.append(date(2014, 6, 1))
StartList.append(date(2014, 9, 1))

EndList.append(date(2014, 3, 1))
EndList.append(date(2014, 6, 1))
EndList.append(date(2014, 9, 1))
EndList.append(date(2014, 12, 1))

Var = 'sst'
Type = '%s_qc_BiasCorrfb_oo_qc_fdbk' % Var

print('--' * 40)
S1 = time.time()

if os.path.isdir('%s\\ErrorEstimation\\Preprocessed' % BaseDir) is False:
    os.makedirs('%s\\ErrorEstimation\\Preprocessed' % BaseDir)
os.chdir('%s\\ErrorEstimation\\Preprocessed' % BaseDir)

if os.path.isdir('Innovations_%s\\2014' % Var) is False:
    os.makedirs('Innovations_%s\\2014' % Var)

# Store the innovations for each day, to be accessed easier in later scripts.
for t in range(dayCount):
    Date = Start + t * dt
    if os.path.isdir('%s\\%s\\as20_obs_files~\\as20_obs_files\\%s' % (BaseDir, PreDir, Start.year)):
        os.chdir('%s\\%s\\as20_obs_files~\\as20_obs_files\\%s' % (BaseDir, PreDir, Start.year))
    else:
        print('"%s\\%s\\as20_obs_files~\\as20_obs_files\\%s" is not available.' % (BaseDir, PreDir, Start.year))
        print('Please ensure correct feedback files are available under Base Directory.')
        exit(0)

    [LAT, LON, OBS] = read_observation(Date, Var)

    os.chdir('%s\\ErrorEstimation\\Preprocessed\\Innovations_%s\\2014' % (BaseDir, Var))
    np.savez(('%s' % Date), xi=LON, yi=LAT, zi=OBS)
print('%s : %s complete.' % (Start, End))

os.chdir('%s\\ErrorEstimation\\Preprocessed' % BaseDir)
for Name in os.listdir('Innovations_%s\\2014' % Var):
    if Name[5:7] == '12':
        copyfile('Innovations_%s\\2014/%s' % (Var, Name), 'Innovations_%s\\2013\\%s' % (Var, Name))

if os.path.isdir('Innovations_%s\\2013' % Var) is False:
    os.makedirs('Innovations_%s\\2013' % Var)
os.chdir('Innovations_%s\\2013' % Var)
for Name in os.listdir('.'):
    if Name[0:4] == '2013':
        os.remove(Name)
    else:
        os.rename(Name, Name.replace('2014', '2013'))

print('Innovations have been successfully stored in ErrorEstimation\\Preprocessed\\Innovations_%s. %4.2f s'
      % (Var, time.time() - S1))
print('--' * 40)

# Prepare Rossby radius.
os.chdir('%s\\%s' % (BaseDir, PreDir))
R = Dataset('rossby_radii.nc', 'r')
ross = np.array(R.variables['rossby_r'][:])
rlat = np.array(R.variables['lat'][:])
rlon = np.array(R.variables['lon'][:])
R.close()

rlat = rlat[np.where(np.array(ross) <= 200000)]
rlon = rlon[np.where(np.array(ross) <= 200000)]
Ross = ross[np.where(ross <= 200000)]

# Prepare TGradient dimensions and mask.
os.chdir('rose_as20_assim_mo\\diaopfoam_files_2013_DEC\\dec')
DIA = Dataset('20131201T0000Z_diaopfoam_fcast.grid_T/20131201T0000Z_diaopfoam_fcast.grid_T.nc')
Depth = np.array(DIA.variables['deptht_bounds'][:])
Mask = ma.getmask(ma.masked_equal(np.array(DIA.variables['votemper'][0,:,:,:]), 0))
DIA.close()
#

# Make the seasonal coarse grids.
for Start, End in zip(StartList, EndList):
    dayCount = (End - Start).days
    InnovationCount = 0
    Count = 0
    dayIdx = np.zeros(dayCount + 1, np.int64)

    if Start.month == 12:
        season = '1-DJF'
    elif Start.month == 3:
        season = '2-MAM'
    elif Start.month == 6:
        season = '3-JJA'
    elif Start.month == 9:
        season = '4-SON'
    else:
        season = 'fail'

    os.chdir('%s\\ErrorEstimation\\Preprocessed' % BaseDir)

    for t in range(dayCount):
        Date = Start + t * dt
        cg = np.load('Innovations_%s\\%s\\%s.npz' % (Type[:3], Date.year, Date))
        dayIdx[t] = InnovationCount
        InnovationCount += cg['xi'].size
    dayIdx[dayCount] = InnovationCount

    SST = np.zeros((InnovationCount, 1))
    LAT = np.zeros(InnovationCount)
    LON = np.zeros(InnovationCount)
    dayNum = np.zeros(InnovationCount, np.int)

    for t in range(dayCount):
        Date = Start + t * dt
        S2 = time.time()

        cg = np.load('Innovations_%s\\%s\\%s.npz' % (Type[:3], Date.year, Date))
        Xi, Yi, Zi = cg['xi'], cg['yi'], cg['zi']

        dataSize = Xi.size
        SST[Count:Count + dataSize] = Zi
        LAT[Count:Count + dataSize] = Yi
        LON[Count:Count + dataSize] = Xi
        dayNum[Count:Count + dataSize] = t
        Count += dataSize

    os.chdir('%s' % season)
    latIdx = np.digitize(LAT, yedges) - 1
    lonIdx = np.digitize(LON, xedges) - 1

    Val = np.zeros((len(yedges) - 1, len(xedges) - 1), np.float)

    GriddedValues = griddata(dayCount, dayIdx, latIdx.flatten(), lonIdx.flatten(), SST.flatten(), Val)
    zi = np.ma.masked_invalid(GriddedValues)

    zi2 = np.zeros([len(yedges) - 1, len(xedges) - 1])
    for i in range(0, len(yedges) - 1, 1):
        for j in range(0, len(xedges) - 1, 1):
            zi2[i, j] = np.mean(zi[i, j, :])
    zi2[zi2 == 0] = np.nan
    np.savez('coarse_grid_%s' % Type[:3], xi=xedges, yi=yedges, zi=zi2)
    print('Coarse grid have successfully been stored in ErrorEstimation\\Preprocessed\\%s. %4.2f s'
          % (season, time.time() - S1))

    # Make the seasonal Rossy Radius npy file.
    gridz = np.ma.masked_invalid(zi2)
    ross = interpolate.griddata((rlon, rlat), np.array(Ross), (x2, y2), 'nearest')
    ross[gridz.mask == True] = 0
    np.save('rossby', ross)
    print(
        'Rossby radius on the coarse grid has been stored successfully in ErrorEstimation\\Preprocessed\\%s. %4.2f s'
        % (season, time.time() - S1))

    # Make the seasonal TGradient (the depth profile for length-scale ratio model file).
    T = np.zeros((Mask.shape[0], Mask.shape[1], Mask.shape[2]))
    os.chdir('%s\\%s\\rose_as20_assim_mo' % (BaseDir, PreDir))
    for t in range(dayCount):
        Date = Start + t*dt
        Year = Date.year
        Month = calendar.month_abbr[Date.month]
        YYYYMMDD = '%i%02i%02i' % (Year, Date.month, Date.day)
        Dir = 'diaopfoam_files_%i_%s\\%s\\' % (Year, Month.upper(), Month.lower())
        X = Dataset('%s%sT0000Z_diaopfoam_fcast.grid_T\\%sT0000Z_diaopfoam_fcast.grid_T.nc' % (Dir, YYYYMMDD, YYYYMMDD), 'r')

        tmp = np.array(X.variables['votemper'][0, :, :, :])
        X.close()
        for x in range(tmp.shape[0]-1):
            T[x, :, :] += 10*(tmp[x, :, :] - tmp[x+1, :, :])/(Depth[x+1, 0]-Depth[x, 0])

    T = T/dayCount
    T = abs(T)
    T[T > 1.5] = 1.5
    T[T < 0.07] = 0.07
    for x in np.linspace(0, 10, 11):
        tval = np.where(T[int(x), :, :] < 0.5)
        for i in range(0, len(tval[0])):
            T[int(x), tval[0][i], tval[1][i]] = 0.5

    for i in range(T.shape[1]):
        for j in range(T.shape[2]):
            T[:, i, j] = smooth_var_array(T[:, i, j], 2)

    os.chdir('%s\\ErrorEstimation\\Preprocessed\\%s' % (BaseDir, season))

    T = ma.masked_array(T, Mask)
    I = np.ones_like(T)
    if os.path.isfile('TGradient.nc') is False:
        copyfile('%s\\Prerequisites\\Bmod_sdv_wgt_rea_t.nc' % BaseDir, 'TGradient.nc')
    P = Dataset('TGradient.nc', 'r+')
    P['wgt2'][:] = T/1.5
    P['wgt1'][:] = (I - T/1.5)
    P.close()
    print('The depth gradient files have been produced successfully from model data. %4.2f s' % (time.time() - S1))

    # Make the sub-level directories for future runs.
    os.chdir('%s\\ErrorEstimation' % BaseDir)
    if os.path.isdir('%s' % Var.upper()) is False:
        os.makedirs('%s' % Var.upper())
    os.chdir('%s' % Var.upper())

    if os.path.isdir('%s' % season) is False:
        os.makedirs('%s' % season)
    os.chdir('%s' % season)
    for N in [100 / 100, 100 / 90, 100 / 70, 100 / 50, 100 / 30, 100 / 10, 100 / 1]:
        if os.path.isdir('%i' % (100 / N)) is False:
            os.makedirs('%i' % (100 / N))
    print('Required subdirectories successfully created. %4.2f s' % (time.time() - S1))
    print('--' * 40)
