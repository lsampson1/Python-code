import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt


N = [1000]
C = [1000]
COV = False
COR = False
RAW = True
Bias = False
cova = []
corr = []
vari = []

a=0
b=0.7
Bscale = 0  # 0 equals no bias

tmp=0

if Bscale != 0:
    Bias = True

for n, c in zip(N, C):
    rand = np.zeros((c, n))
    if Bias:
        U = np.exp((-np.linspace(0, Bscale, c)**2)/(2*(4**2)))  # Adding bias function
        d = -0.5
        e = a+(1/b)
    else:
        U = 0  # No bias
        e = a+1/b
        d = a

    for T in range(n):
        nr = 1-2*np.random.rand(1)[0]
        rand[:, T] = U*nr + (1-U)*expon.rvs(loc=a, scale=b, size=int(c))
        if np.max(rand[:,T]) > tmp:
            tmp = np.max(rand[:,T])

    rand2 = (rand.T-np.mean(rand, axis=1)).T
    rand2 = np.mean(rand2*rand2[0, :], axis=1)
    vari.append(np.mean(rand, axis=1))
    cova.append(rand2)
    corr.append(rand2 / (np.sqrt(np.mean(rand**2, axis=1)-np.mean(rand, axis=1)**2)*np.sqrt(np.mean(rand[0, :]**2)-np.mean(rand[0, :])**2)))

weights = np.zeros_like(cova)
colors = ['b', 'g', 'orange', 'skyblue', 'lightgreen', 'wheat', 'navy', 'darkgreen', 'red']
cs = colors[:len(cova)]
cs2 = colors[len(cova):2*len(cova)]

if COV is True:
    y = []
    fig1, axs1 = plt.subplots(2)
    fig1.suptitle('Covariance (without r = 0)')
    label = []
    print(40*'--')
    print('Covariance diagnostics. \n')
    for n in range(len(cova)):
        y.append(np.linspace(0, np.max(C), C[n]))
        axs1[0].plot(y[n], cova[n], c=cs2[n], linestyle='--')
        weights[n] = np.ones_like(cova[n]) / float(len(cova[n][1:]))
        label.append('N = %s, C = %s.' % (N[n], C[n]))
        print('N = %s, C = %s :  Mean = %6.4f ' % (N[n], C[n], np.nanmean(cova[n])))

    axs1[1].set_title('Histogram - Cov')
    bins = np.array(np.linspace(-e/10, e/10, 20)).T
    axs1[1].hist(cova, bins=bins, weights=list(weights), color=cs, label=label)
    axs1[0].label_outer()

if COR is True:
    y = []
    fig2, axs2 = plt.subplots(2)
    fig2.suptitle('Correlation')
    label = []
    print(40*'--')
    print('Correlation diagnostics. \n')
    for n in range(len(corr)):
        y.append(np.linspace(0, np.max(C), C[n]))
        axs2[0].plot(y[n], corr[n], c=cs2[n], linestyle='--')
        weights[n] = np.ones_like(corr[n]) / float(len(corr[n]))
        label.append('N = %s, C = %s.' % (N[n], C[n]))
        print('N = %s, C = %s :  Mean = %6.4f ' % (N[n], C[n], np.nanmean(corr[n])))
    axs2[1].set_title('Histogram - Cor')
    bins = np.array(np.linspace(-1, 1, 20)).T
    axs2[1].hist(corr, bins=bins, weights=list(weights), color=cs, label=label)
    axs2[0].label_outer()

if RAW is True:
    y = []
    fig3, axs3 = plt.subplots(2)
    fig3.suptitle('Raw Data')
    label = []
    print(40*'--')
    print('Data diagnostics. \n')
    for n in range(len(vari)):
        y.append(np.linspace(0, np.max(C), C[n]))
        axs3[0].scatter(y[n], vari[n], c=cs2[n], marker='+')
        weights[n] = np.ones_like(vari[n]) / float(len(vari[n]))
        label.append('N = %s, C = %s.' % (N[n], C[n]))
        print('N = %s, C = %s :  Mean = %6.4f ' % (N[n], C[n], np.nanmean(vari[n])))
    axs3[1].set_title('Histogram - Raw')
    bins = np.array(np.linspace(d, e, 20)).T
    axs3[1].hist(vari, bins=bins, weights=list(weights), color=cs, label=label)
    axs3[0].label_outer()
plt.legend()
plt.show()
print(1)