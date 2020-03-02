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
p = 0

Seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]
Gridded = False

if Gridded:
    ipa_str1 = 'Data\\IPA_Sdv_Grid.npy'
    hl_str1 = 'Data\\HL_Sdv_Grid.npy'
    ipa_str2 = 'Data\\IPA_Lsr_Grid.npy'
    hl_str2 = 'Data\\HL_Lsr_Grid.npy'
else:
    ipa_str1 = 'Data\\IPA_Sdv_Real.npy'
    hl_str1 = 'Data\\HL_Sdv_Real.npy'
    ipa_str2 = 'Data\\IPA_Lsr_Real.npy'
    hl_str2 = 'Data\\HL_Lsr_Real.npy'

os.chdir(r'\\POFCDisk1\PhD_Lewis\ErrorEstimation\Preprocessed')
nlist = [100/100, 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]

print('Nan values')
print('--'*40)
dt = datetime.timedelta(hours=24)
fig, ax = plt.subplots(2, figsize=(8, 6))
Typ = 'sst'

tipa = []
tlsr = []
thl = []
thlr = []

for Season in Seasonlist:
    ipa = []
    lsr = []
    hl = []
    hlr = []
    print('Season = %s' % Season)
    os.chdir(r'\\POFCDisk1\PhD_Lewis\ErrorEstimation\Preprocessed\%s' % Season)
    GRID = np.load('coarse_grid_%s.npz' % Typ)
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

    os.chdir(r'\\POFCDisk1\PhD_Lewis\ErrorEstimation\%s\%s' % (Typ, Season))
    TRUIPA = np.sqrt(np.load(r"100\%s" % ipa_str1))
    TRUHL = np.sqrt(np.load(r"100\%s" % hl_str1))
    TRULSR = np.load(r"100\%s" % ipa_str2)
    TRUHLR = np.load(r"100\%s" % hl_str2)

    for M in [TRUHLR, TRULSR]:
        M[M > 1] = np.nan
        M[M < 0] = np.nan

    Tru = np.sum(gridz.mask == True)
    TMean = [np.nanmean(TRUIPA), np.nanmean(TRUHL), np.nanmean(TRULSR), np.nanmean(TRUHLR)]
    TVar = [np.nanvar(TRUIPA), np.nanvar(TRUHL), np.nanvar(TRULSR), np.nanvar(TRUHLR)]

    #
    # TRUIPA = inter(TRUIPA, points)
    # TRULSR = inter(TRULSR, points)
    # TRUHL = inter(TRUHL, points)
    # TRUHLR = inter(TRUHLR, points)

    for N in nlist:  # Use every Nth observation. (N=1 is every observation)
        os.chdir(r'\\POFCDisk1\PhD_Lewis\ErrorEstimation\%s\%s\%i' % (Typ, Season, 100 / N))
        if os.path.isfile('%s' % ipa_str1) is False:
            continue
        elif os.path.isfile('%s' % hl_str1) is False:
            continue
        else:
            IPA = np.sqrt(np.load('%s' % ipa_str1))
            LSR = np.load('%s' % ipa_str2)
            HL = np.sqrt(np.load('%s' % hl_str1))
            HLR = np.load('%s' % hl_str2)

        for M in [LSR, HLR]:
            M[M > 1] = np.nan
            M[M < 0] = np.nan

        Var = [np.nanvar(IPA), np.nanvar(HL), np.nanvar(LSR), np.nanvar(HLR)]
        Mean = [np.nanmean(IPA), np.nanmean(HL), np.nanmean(LSR), np.nanmean(HLR)]
        Covar = [np.nanmean((IPA-Mean[0])*(TRUIPA-TMean[0])), np.nanmean((HL-Mean[1])*(TRUHL-TMean[1])),
                 np.nanmean((LSR-Mean[2])*(TRULSR-TMean[2])), np.nanmean((HLR-Mean[3])*(TRUHLR-TMean[3]))]

        ipa.append(1-(2*Covar[0]/(Var[0]+TVar[0] + (Mean[0]-TMean[0])**2)))
        hl.append(1-(2*Covar[1]/(Var[1]+TVar[1] + (Mean[1]-TMean[1])**2)))
        lsr.append(1-(2*Covar[2]/(Var[2]+TVar[2] + (Mean[2]-TMean[2])**2)))
        hlr.append(1-(2*Covar[3]/(Var[3]+TVar[3] + (Mean[3]-TMean[3])**2)))

        ESIPA = np.sum(np.isnan(IPA))
        ESLSR = np.sum(np.isnan(LSR))
        ESHL = np.sum(np.isnan(HL))
        ESHLR = np.sum(np.isnan(HLR))

        EI = (ESIPA - Tru)
        EL1 = (ESLSR - Tru)
        EH = (ESHL - Tru)
        EL2 = (ESHLR - Tru)
        print('IPA: (%3.0f, %3.0f), HL: (%3.0f, %3.0f), Percentage = %3.0f ' % (EI, EL1, EH, EL2, 100/N))

    cs = ['deepskyblue', 'forestgreen', 'gold', 'sandybrown']
    ax[0].plot((100/np.array(nlist)), ipa, c=cs[p], label=Season)
    ax[0].plot((100/np.array(nlist)), hl, c=cs[p], linestyle=':')
    ax[0].plot([0])

    ax[1].plot((100/np.array(nlist)), lsr, c=cs[p])
    ax[1].plot((100/np.array(nlist)), hlr, c=cs[p], linestyle=':')
    fig.text(0.5, 0.01, 'Percentage of observations', ha='center')
    fig.text(0.01, 0.5, 'Discordance', va='center', rotation='vertical')
    ax[0].legend()

    tipa.append(ipa)
    thl.append(hl)
    tlsr.append(lsr)
    thlr.append(hlr)
    p += 1
fig.suptitle('Seasonal averages')
ax[0].set_title('Standard deviation')
ax[1].set_title('Length-scale ratio')
ax[1].plot([0], c='k', label='IPA')
ax[1].plot([0], c='k', linestyle=':', label='HL')
ax[1].legend()
plt.show()

plt.figure('Average concordence - SDV')
plt.plot((100/np.array(nlist)), np.mean(tipa, axis=0), c='k', label='IPA')
plt.plot((100/np.array(nlist)), np.mean(thl, axis=0), c='k', linestyle=':', label='HL')
plt.ylim(0, 1)
plt.ylabel('Discordance')
plt.xlabel('Percentage of observations')
plt.legend()

plt.figure('Average discordance - LSR')
plt.plot((100/np.array(nlist)), np.mean(tlsr, axis=0), c='k', label='IPA')
plt.plot((100/np.array(nlist)), np.mean(thlr, axis=0), c='k', linestyle=':', label='HL')
plt.ylim(0, 1)
plt.ylabel('Discordance')
plt.xlabel('Percentage of observations')
plt.legend()
plt.show()
