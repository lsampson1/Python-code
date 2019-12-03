import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

Seasonlist = ["1-DJF"]

dt = datetime.timedelta(hours=24)
for N in [1]:  # Use every Nth observation. (N=1 is every observation)
    size = np.zeros((len(Seasonlist)))

    for Season in Seasonlist:
        zi = []
        yi = []
        xi = []

        if Season == '1-DJF':
            Start = datetime.date(2013, 12,  1)
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

        for t in range(dayCount):
            date = Start + t*dt
            # Open and save all innovations,

            dat = np.load('%i/%04i-%02i-%02i.npz' % (date.year, date.year, date.month, date.day))

            idx = np.round(dat['xi'], 1) == 63.000

            tmpy = dat['yi'][idx]
            tmpz = dat['zi'][idx, 0]
            tmpx = dat['xi'][idx]

            zi.append(np.array([x for _, x in sorted(zip(tmpy, tmpz))]))
            xi.append(np.array([x for _, x in sorted(zip(tmpy, tmpx))]))
            yi.append(np.array([x for _, x in sorted(zip(tmpy, tmpy))]))
            dat.close()

bin = np.linspace(8, 32, 121)
bin_means = []
bin_center = 0.5 * (bin[1:] + bin[:-1])


for d in range(len(yi)):
    digitized = np.digitize(yi[d], bin)
    bin_means.append([zi[d][digitized == i].mean() for i in range(1, len(bin))])

tmp = np.array(bin_means)

cov = np.nanmean((tmp - np.nanmean(tmp, axis=0)).T*(tmp[:, 60] - np.nanmean(tmp[:, 60])), axis=1)

corr = cov / (np.sqrt(np.nanmean(tmp**2, axis=0)-np.nanmean(tmp, axis=0)**2)*np.sqrt(np.nanmean(tmp[:, 60]**2)-np.nanmean(tmp[:, 60])**2))

plt.figure(1)
data = np.nanmean(tmp, axis=0)
tmpdata = data.copy()
tmpdata[np.isnan(tmpdata)] = 0

plt.plot(bin_center, data, label='Data (mean)')
plt.plot(bin_center, cov, label='Covariance')
plt.plot(bin_center, corr, label='Correlation')
plt.title('Innovation statistics (y-Hx)')
plt.xlabel('latitude (degrees)')
plt.legend()

plt.figure(2)
plt.hist([data[~np.isnan(data)], cov[~np.isnan(cov)], corr[~np.isnan(corr)]], 20, range=(-1, 1),
         label=['Data (mean)', 'Covariance', 'Correlation'])
plt.title('Density of distribution')
plt.legend()
plt.show()
