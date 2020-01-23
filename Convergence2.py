import numpy as np
import os
import datetime
import numpy.ma as ma
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


def inter(px, ps):
    p2 = px[~np.isnan(px)]

    ps2 = np.zeros((2, len(p2)))
    ps2[0, :] = ps[0, (~np.isnan(px)).flatten()]
    ps2[1, :] = ps[1, (~np.isnan(px)).flatten()]

    px = interpolate.griddata(ps2.T, p2.flatten(), (x1, y1), 'nearest')
    px[gridz.mask == True] = np.nan
    return px


np.seterr(all='ignore')
# Gaussian projection length-scale
a0 = 0.25
a1 = 4

Seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]
Gridded = True

if Gridded:
    ipa_str1 = 'Data\\IPA_Sdv_Grid.npy'
    hl_str1 = 'Data\\HL_Sdv_Grid.npy'
    ipa_str2 = 'Data\\IPA_Lsr_Grid.npy'
    hl_str2 = 'Data\\HL_Lsr_Grid.npy'
else:
    ipa_str1 = 'Data\\IPA_Sdv_Real.npy'
    hl_str1 = 'Data\\HL_Sdv_Grid.npy'
    ipa_str2 = 'Data\\IPA_Lsr_Real.npy'
    hl_str2 = 'Data\\HL_Lsr_Grid.npy'

os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed')
nlist = [100/100, 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]

dt = datetime.timedelta(hours=24)
fig, ax = plt.subplots(2, figsize=(8, 6))
plt.title('test')

NAN = ([], [], [])
tripa, trlsr, tgipa, tglsr, thl, thlr = [], [], [], [], [], []
m = 0
ls = ['-', '--', ':']
cs = ['deepskyblue', 'forestgreen', 'gold', 'sandybrown']
namelist = [('IPA', 'real'), ('IPA', 'grid'), ('HL', 'grid')]
Typ = 'sst'

for name in namelist:
    str1 = 'Data\\%s_Sdv_%s.npy' % (name[0], name[1])
    str2 = 'Data\\%s_Lsr_%s.npy' % (name[0], name[1])
    p = 0
    for Season in Seasonlist:
        sdv = []
        lsr = []

        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\%s' % Season)
        GRID = np.load('coarse_grid_sst.npz')
        gridz = GRID['zi']
        x2 = GRID['xi']
        y2 = GRID['yi']
        gridz = ma.masked_invalid(gridz)
        GRID.close()

        x1 = np.zeros((len(x2) - 1, len(y2) - 1))
        for i in range(0, len(y2) - 1, 1):
            x1[:, i] = x2[:-1]
        x1 = np.transpose(x1)
        y1 = np.zeros((len(y2) - 1, len(x2) - 1))
        for i in range(0, len(x2) - 1, 1):
            y1[:, i] = y2[:-1]
        points = np.array((x1.flatten(), y1.flatten()))

        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s\%s' % (Typ.upper(), Season))
        TRUSDV = np.load(r"True\%s" % str1)
        TRULSR = np.load(r"True\%s" % str2)

        TRUSDV = inter(TRUSDV, points)
        TRULSR = inter(TRULSR, points)

        for N in nlist:  # Use every Nth observation. (N=1 is every observation)
            os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s\%s\%i' % (Typ, Season, 100 / N))
            if os.path.isfile('%s' % str1) is False:
                continue
            else:
                SDV = (np.load(str1))
                LSR = np.load(str2)

            SDV = inter(SDV, points)
            LSR = inter(LSR, points)

            sdver = np.nanmean(abs(SDV - TRUSDV))
            lsrer = np.nanmean(abs(LSR - TRULSR))
            sdv.append(100*sdver/(np.nanmean(TRUSDV)+sdver))
            lsr.append(100*lsrer/(np.nanmean(TRULSR)+lsrer))

            ESIPA = np.sum(np.isnan(SDV))
            ESLSR = np.sum(np.isnan(LSR))

            ES = (ESIPA - np.sum(gridz.mask))
            EL = (ESLSR - np.sum(gridz.mask))
            NAN[m].append('%s - %s: (%3.0f, %3.0f), Percentage = %3.0f ' % (name[0], name[1], ES, EL, 100/N))

        ax[0].plot((100/np.array(nlist)), sdv, c=cs[p], linestyle=ls[m])
        ax[0].plot([0])
        ax[0].set_ylabel('Standard deviation')

        ax[1].plot((100/np.array(nlist)), lsr, c=cs[p], linestyle=ls[m])
        ax[1].set_ylabel('Length-scale')
        fig.text(0.5, 0.01, 'Percentage of observations', ha='center')
        fig.text(0.01, 0.5, 'Percentage of corroboration', va='center', rotation='vertical')
        if name[1] == 'real':
            ax[0].plot([0], c=cs[p], label=Season)
            if p == 3:
                ax[0].legend()

        if name[1] == 'real':
            tripa.append(sdv)
            trlsr.append(lsr)
        elif name[0] == 'IPA' and name[1] == 'grid':
            tgipa.append(sdv)
            tglsr.append(lsr)
        elif name[0] == 'HL':
            thl.append(sdv)
            thlr.append(lsr)
        p += 1
    m += 1
for n in range(m):
    ax[1].plot([0], c='k', linestyle=ls[n], label='%s - %s ' % (namelist[n][0], namelist[n][1]))
ax[1].legend()
plt.show()

plt.figure('Average concordence - SDV')
plt.title('Yearly average - SDV')
plt.plot((100/np.array(nlist)), np.mean(tripa, axis=0), c='k', linestyle='-', label='IPA - real')
plt.plot((100/np.array(nlist)), np.mean(tgipa, axis=0), c='k', linestyle='--', label='IPA - grid')
plt.plot((100/np.array(nlist)), np.mean(thl, axis=0), c='k', linestyle=':', label='HL')
plt.ylabel('%')
plt.xlabel('%')
plt.legend()

plt.figure('Average concordence - LSR')
plt.title('Yearly average - LSR')
plt.plot((100/np.array(nlist)), np.mean(trlsr, axis=0), c='k',  linestyle='-', label='IPA - real')
plt.plot((100/np.array(nlist)), np.mean(tglsr, axis=0), c='k',  linestyle='--', label='IPA - grid')
plt.plot((100/np.array(nlist)), np.mean(thlr, axis=0), c='k', linestyle=':', label='HL')
plt.ylabel('%')
plt.xlabel('%')
plt.legend()
plt.show()

print(65*'__')
print('Nan values')
print('=='*65)
for q in range(p-1):
    print('Season - %s' % Seasonlist[q])
    print(65*'--')
    for n in range(len(nlist)):
        print('%s || %s || %s' % (NAN[0][n + q*len(nlist)], NAN[1][n + q*len(nlist)], NAN[2][n + q*len(nlist)]))
