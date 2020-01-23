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

os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed')
nlist = [100/100, 100/90, 100/70, 100/50, 100/30, 100/10, 100/1]

print('Nan values')
print('--'*40)
dt = datetime.timedelta(hours=24)
fig, ax = plt.subplots(2, figsize=(8, 6))
plt.title('test')
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
    os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\Preprocessed\%s' % Season)
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

    os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s\%s' % (Typ, Season))
    TRUIPA = np.sqrt(np.load(r"True\%s" % ipa_str1))
    TRUHL = np.sqrt(np.load(r"True\%s" % hl_str1))
    TRULSR = np.load(r"True\%s" % ipa_str2)
    TRUHLR = np.load(r"True\%s" % hl_str2)

    # for M in [TRUHLR, TRULSR]:
    #     M[M > 1] = np.nan
    #     M[M < 0] = np.nan
    #
    # TRUIPA = inter(TRUIPA, points)
    # TRULSR = inter(TRULSR, points)
    # TRUHL = inter(TRUHL, points)
    # TRUHLR = inter(TRUHLR, points)

    for N in nlist:  # Use every Nth observation. (N=1 is every observation)
        os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics\%s\%s\%i' % (Typ, Season, 100 / N))
        if os.path.isfile('%s' % ipa_str1) is False:
            continue
        elif os.path.isfile('test\\%s' % hl_str1) is False:
            continue
        else:
            IPA = np.sqrt(np.load(ipa_str1))
            LSR = np.load(ipa_str2)
            HL = np.sqrt(np.load('test\\%s' % hl_str1))
            HLR = np.load('test\\%s' % hl_str2)

        for M in [LSR, HLR]:
            M[M>2] = np.nan
            M[M<-1] = np.nan
        # for M in [IPA,LSR,HL,HLR,TRUIPA,TRUHL,TRUHLR,TRULSR]:
        #     M[np.isnan(M)] = 0

        # IPA = inter(IPA, points)
        # LSR = inter(LSR, points)
        # HL = inter(HL, points)
        # HLR = inter(HLR, points)

        ipaer = np.nanmean(abs(IPA - TRUIPA))
        lsrer = np.nanmean(abs(LSR - TRULSR))
        hler = np.nanmean(abs(HL - TRUHL))
        hlrer = np.nanmean(abs(HLR - TRUHLR))
        ipa.append(100*ipaer/(np.nanmean(TRUIPA)+ipaer))
        lsr.append(100*lsrer/(np.nanmean(TRULSR)+lsrer))
        hl.append(100*hler/(np.nanmean(TRUHL)+hler))
        hlr.append(100*hlrer/(np.nanmean(TRUHLR)+hlrer))

        ESIPA = np.sum(np.isnan(IPA))
        ESLSR = np.sum(np.isnan(LSR))
        ESHL = np.sum(np.isnan(HL))
        ESHLR = np.sum(np.isnan(HLR))

        EI = (ESIPA - np.sum(gridz.mask))
        EL1 = (ESLSR - np.sum(gridz.mask))
        EH = (ESHL - np.sum(gridz.mask))
        EL2 = (ESHLR - np.sum(gridz.mask))
        print('IPA: (%3.0f, %3.0f), HL: (%3.0f, %3.0f), Percentage = %3.0f ' % (EI, EL1, EH, EL2, 100/N))

    cs = ['deepskyblue', 'forestgreen', 'gold', 'sandybrown']
    ax[0].plot((100/np.array(nlist)), ipa, c=cs[p], label=Season)
    ax[0].plot((100/np.array(nlist)), hl, c=cs[p], linestyle=':')
    ax[0].plot([0])
    ax[0].set_ylabel('Standard deviation')

    ax[1].plot((100/np.array(nlist)), lsr, c=cs[p])
    ax[1].plot((100/np.array(nlist)), hlr, c=cs[p], linestyle=':')
    ax[1].set_ylabel('Length-scale')
    fig.text(0.5, 0.01, 'Percentage of observations', ha='center')
    fig.text(0.01, 0.5, 'Percentage of change (True-Percent)/(True+Percent)', va='center', rotation='vertical')
    ax[0].legend()

    tipa.append(ipa)
    thl.append(hl)
    tlsr.append(lsr)
    thlr.append(hlr)
    p += 1

ax[1].plot([0], c='k', label='IPA')
ax[1].plot([0], c='k', linestyle=':', label='HL')
ax[1].legend()
plt.show()

plt.figure('Average concordence - SDV')
plt.title('Yearly average - SDV')
plt.plot((100/np.array(nlist)), np.mean(tipa, axis=0), c='k', label='IPA')
plt.plot((100/np.array(nlist)), np.mean(thl, axis=0), c='k', linestyle=':', label='HL')
plt.ylabel('%')
plt.xlabel('%')
plt.legend()

plt.figure('Average concordence - LSR')
plt.title('Yearly average - LSR')
plt.plot((100/np.array(nlist)), np.mean(tlsr, axis=0), c='k', label='IPA')
plt.plot((100/np.array(nlist)), np.mean(thlr, axis=0), c='k', linestyle=':', label='HL')
plt.ylabel('%')
plt.xlabel('%')
plt.legend()
plt.show()
