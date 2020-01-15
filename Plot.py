import numpy as np
import os
import shutil
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from netCDF4 import Dataset


def cgrid2mgrid(variable, sigma, dst, gridx2, gridy2, typ):

    if typ == 'sst':
        label = 'Temperature'
        ll = 0.2
        ul = 0.8
    elif typ == 'sla':
        label = 'Sea level anomoly'
        ll = 0
        ul = 0.3
    else:
        label = 'False'
        ll = 0
        ul = 1
    cwdi = os.getcwd()
    os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed')

    mo = Dataset('S010_1d_20111130_20111201_gridT.nc', 'r')
    mo2 = Dataset('Bmod_sdv_mxl_rea_t.nc', 'r')
    obs = Dataset('sst_err_obs_sd.nc', 'r')

    lat = np.array(mo.variables['nav_lat'])
    lon = np.array(mo.variables['nav_lon'])
    mo.close()

    mod = mo2.variables['sdv_mxl'][:]
    mo2.close()

    omod = obs.variables['Err_rep_SST_sig'][:]
    obs.close()

    os.chdir(cwdi)

    point = np.array((gridx2.flatten(), gridy2.flatten()))

    if dst[-17:-14] == 'Sdv':
        data = smooth_var_array(variable, sigma)
        data = np.array(data.flatten())

        griddz = interpolate.griddata(point.T, data, (lon, lat), 'linear')

        griddz = ma.masked_array(griddz, mask=mod.mask)
        nnz = griddz[~np.isnan(griddz)]

        plt.figure()
        plt.title('Background error standard deviation')
        plt.pcolormesh(griddz, cmap='jet', vmin=ll, vmax=ul)
        plt.colorbar()
        plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(nnz, 5), np.percentile(nnz, 95)))
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig('%s.png' % (dst[:-3]))
        plt.close()

        # shutil.copyfile(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\Bmod_sdv_mxl_rea_t.nc', 'Data\%s' % dst)
        #
        # sdv = Dataset('Data\%s' % dst, 'r+')
        # sdv['sdv_mxl'][:] = griddz
        # sdv.close()

    if dst[-17:-14] == 'Obs':
        data = smooth_var_array(variable, sigma)
        data = np.array(data.flatten())

        griddz = interpolate.griddata(point.T, data, (lon, lat), 'linear')

        griddz = ma.masked_array(abs(griddz), mask=omod.mask)
        nnz = griddz[~np.isnan(griddz)]

        plt.figure()
        plt.title('Observation error standard deviation')
        plt.pcolormesh(griddz, cmap='jet', vmin=0, vmax=0.5)
        plt.colorbar()
        plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(nnz, 5), np.percentile(nnz, 95)))
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig('%s.png' % (dst[:-3]))
        plt.close()
        #
        # shutil.copyfile(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\sst_err_obs_sd.nc', 'Data\%s' % dst)
        #
        # sdv = Dataset('Data\%s' % dst, 'r+')
        # sdv['Err_rep_SST_sig'][:] = np.sqrt(griddz)
        # sdv.close()

    if dst[-17:-14] == 'Lsr':
        variable[variable < 0] = 0
        variable[variable > 1] = 1

        data = smooth_var_array(variable, 2*sigma)
        data = np.array(data.flatten())

        griddz = interpolate.griddata(point.T, data, (lon, lat), 'linear')

        griddz = ma.masked_array(griddz, mask=mod.mask)
        nnz = griddz[~np.isnan(griddz)]

        plt.figure()
        plt.title('Short length-scale ratio')
        plt.pcolormesh(griddz, cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(nnz, 5), np.percentile(nnz, 95)))
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig('%s.png' % (dst[:-3]))
        plt.close()

        # w1 = griddz.copy()
        #
        # w1 = ma.masked_array(w1, mod.mask)
        # w2 = ma.masked_array(1-w1, mod.mask)
        #
        # p = Dataset(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\%s\TGradient.nc' % sea, 'r')
        # w2t = w2 * (p['wgt2'][:])
        # w1t = w1 * (p['wgt1'][:])
        # p.close()
        #
        # wgt2 = w2t / (w2t + w1t)
        #
        # shutil.copyfile(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\Bmod_sdv_wgt_rea_t.nc', 'Data\%s' % dst)
        #
        # r = Dataset('Data\%s' % dst, 'r+')
        # r['wgt2'][:] = np.sqrt(wgt2)
        # r['wgt1'][:] = np.sqrt(1 - wgt2)
        # r.close()

    return


def smooth_var_array(data, sigma):

    data[data == 0] = np.nan
    data = ma.masked_invalid(data)
    masks = 1 - data.mask.astype(np.float)
    wsum = ndimage.gaussian_filter(masks*data, sigma)
    gsum = ndimage.gaussian_filter(masks,      sigma)

    gfilter = wsum/gsum

    gfilter = ma.masked_array(gfilter, mask=data.mask)

    return gfilter


os.chdir(r'\\POFCDisk1\PhD_Lewis\H-L_Variances\Innovations\1-DJF')
Typ = 'sst'
if Typ == 'sst':
    ll = 0.2
    ul = 0.8
elif Typ == 'sla':
    ll = 0
    ul = 0.3
else:
    ll = 0
    ul = 1

# Load coarse grid for domain and mask
GRID = np.load('coarse_grid_%s.npz' % Typ)
gridX1 = GRID['xi']
gridY1 = GRID['yi']
gridZ1 = GRID['zi']
gridZ1 = ma.masked_invalid(gridZ1)
mask = gridZ1.mask
GRID.close()

gridX2 = np.zeros((len(gridX1) - 1, len(gridY1) - 1))
for i in range(0, len(gridY1) - 1, 1):
    gridX2[:, i] = gridX1[:-1]
gridX2 = np.transpose(gridX2)
gridY2 = np.zeros((len(gridY1) - 1, len(gridX1) - 1))
for i in range(0, len(gridX1) - 1, 1):
    gridY2[:, i] = gridY1[:-1]

BiasCorr = False
Gridded = True
nlist = [100/100]  # , 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]

for N in nlist:
    if Gridded:
        name = 'Grid'
    else:
        name = 'Real'
    for Type in ['HL']:
        for Season in ['1-DJF', '2-MAM', '3-JJA', '4-SON']:
            os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s' % Typ.upper())
            # os.chdir(r'%s\%i' % (Season, 100/N))  # For final analysis
            os.chdir(r'%s\True' % Season)  # For 'truth' analysis. nlist must be [100/100] only.
            # os.chdir(r'%s\%i\test' % (Season, 100/N))  # For test analysis

            if os.path.isfile('Data\\%s_Lsr_%s.npy' % (Type, name)):
                x = np.load('Data\\%s_Lsr_%s.npy' % (Type, name))
                x[np.where(x == 0)] = np.nan
                x2 = x[~np.isnan(x)]

                plt.figure()
                plt.pcolormesh(gridX1, gridY1, x, cmap='jet', vmin=0, vmax=1)
                plt.title('Length-scale ratio (wgt1)')
                plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(x2, 5), np.percentile(x2, 95)))
                plt.colorbar()
                plt.tight_layout()
                plt.savefig('%s_Lsr_%s.jpg' % (Type, name))
                plt.close()

                numval = len(x2)

                LimLSR = x.copy()
                lim = sorted(range(len(x2)), key=lambda m: x2[m], reverse=True)[:int(2 * numval / 100)]
                lim += sorted(range(len(x2)), key=lambda m: x2[m], reverse=False)[:int(2 * numval / 100)]

                for j in lim:
                    (u, v) = np.where(x == x2[j])
                    LimLSR[x == x2[j]] = np.nan

                X2 = LimLSR[~np.isnan(LimLSR)]
                points2 = np.zeros((2, len(X2)))
                points = np.array((gridX2.flatten(), gridY2.flatten()))
                points2[0, :] = points[0, (~np.isnan(LimLSR)).flatten()]
                points2[1, :] = points[1, (~np.isnan(LimLSR)).flatten()]

                LimLSR = interpolate.griddata(points2.T, X2.flatten(), (gridX2, gridY2), 'nearest')

                LimLSR[mask == True] = np.nan

                cgrid2mgrid(LimLSR, 1.2, '%s_Lsr_%s_Model.nc' % (Type, name), gridX2, gridY2, Typ)
                # if N == 1.00:
                #     shutil.copyfile('%s_Lsr_Model.nc' % Type, r'\\POFCDisk1\PhD_Lewis\ocean\OPERATIONAL_SUITE_V5.3\
                #     AS20v28\ratio_%s\others\%s_lsr_%s.nc' % (Season[2:].lower(), Type, Typ))

            if os.path.isfile('Data\\%s_Sdv_%s.npy' % (Type, name)):
                y = np.sqrt(np.load('Data\\%s_Sdv_%s.npy' % (Type, name)))
                y[np.where(y == 0)] = np.nan
                y2 = y[~np.isnan(y)]

                plt.figure()
                plt.pcolormesh(gridX1, gridY1, y, cmap='jet', vmin=ll, vmax=ul)
                plt.title('Temperature (degrees)')
                plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(y2, 5), np.percentile(y2, 95)))
                plt.colorbar()
                plt.tight_layout()
                plt.savefig('%s_Sdv_%s.jpg' % (Type, name))
                plt.close()

                numval = len(y2)

                LimSTD = y.copy()
                lim = sorted(range(len(y2)), key=lambda m: y2[m], reverse=True)[:int(2 * numval / 100)]
                lim += sorted(range(len(y2)), key=lambda m: y2[m], reverse=False)[:int(2 * numval / 100)]

                for j in lim:
                    (u, v) = np.where(y == y2[j])
                    LimSTD[y == y2[j]] = np.nan

                Y2 = LimSTD[~np.isnan(LimSTD)]
                points = np.array((gridX2.flatten(), gridY2.flatten()))
                points2 = np.zeros((2, len(Y2)))
                points2[0, :] = points[0, (~np.isnan(LimSTD)).flatten()]
                points2[1, :] = points[1, (~np.isnan(LimSTD)).flatten()]

                LimSTD = interpolate.griddata(points2.T, Y2.flatten(), (gridX2, gridY2), 'nearest')

                LimSTD[mask == True] = np.nan

                cgrid2mgrid(LimSTD, 1.2, '%s_Sdv_%s_Model.nc' % (Type, name), gridX2, gridY2, Typ)
                # if N == 1.00:
                #     shutil.copyfile('%s_Sdv_%s_Model.nc'  % (Type, name), r'\\POFCDisk1\PhD_Lewis\ocean\OPERATIONAL_
                #     SUITE_V5.3\AS20v28\errorcovs_%s\others\%s_sdv_%s.nc' % (Season[2:].lower(), Type, Typ))

            if os.path.isfile('Data\\%s_Obs_%s.npy' % (Type, name)):
                y = np.sqrt(np.load('Data\\%s_Obs_%s.npy' % (Type, name)))
                y[np.where(y == 0)] = np.nan
                y2 = y[~np.isnan(y)]

                plt.figure()
                plt.pcolormesh(gridX1, gridY1, y, cmap='jet', vmin=0.1, vmax=0.5)
                plt.title('Temperature (degrees)')
                plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(y2, 5), np.percentile(y2, 95)))
                plt.colorbar()
                plt.tight_layout()
                plt.savefig('%s_Obs_%s.jpg' % (Type, name))
                plt.close()

                numval = len(y2)

                LimSTD = y.copy()
                lim = sorted(range(len(y2)), key=lambda m: y2[m], reverse=True)[:int(2 * numval / 100)]
                lim += sorted(range(len(y2)), key=lambda m: y2[m], reverse=False)[:int(2 * numval / 100)]

                for j in lim:
                    (u, v) = np.where(y == y2[j])
                    LimSTD[y == y2[j]] = np.nan

                Y2 = LimSTD[~np.isnan(LimSTD)]
                points = np.array((gridX2.flatten(), gridY2.flatten()))
                points2 = np.zeros((2, len(Y2)))
                points2[0, :] = points[0, (~np.isnan(LimSTD)).flatten()]
                points2[1, :] = points[1, (~np.isnan(LimSTD)).flatten()]

                LimSTD = interpolate.griddata(points2.T, Y2.flatten(), (gridX2, gridY2), 'nearest')

                LimSTD[mask == True] = np.nan

                cgrid2mgrid(LimSTD, 1.2, '%s_Obs_%s_Model.nc' % (Type, name), gridX2, gridY2, Typ)
                # if N == 1.00:
                #     shutil.copyfile('%s_Obs_%s_Model.nc' % (Type, name), r'\\POFCDisk1\PhD_Lewis\ocean\OPERATIONAL_
                #     SUITE_V5.3\AS20v28\errorcovs_%s\others\%s_obs_%s.nc' % (Season[2:].lower(), Type, Typ))
