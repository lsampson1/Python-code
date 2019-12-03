import os
import re
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'\\POFCDisk1\PhD_Lewis\EEDiagnostics')

for name in ['Real','Gridded']:
    for typ in ['HL','IPA']:
        x = []
        y = []
        z = []
        if os.path.isfile('%s Diagnostics-%s.txt' % (typ, name)) is False:
            continue
        for txt in open('%s Diagnostics-%s.txt' % (typ, name)):
            try:
                X = (re.findall(r"\d+.\d+", txt))
                x.append(float(X[0]))
                y.append(X[1])
                z.append(X[2])
            except:
                continue
        ploty = np.array(0)
        plotx = np.array(0)
        for i in range(int(len(y) / 4)):
            ploty = np.append(ploty, y[i * 4])
            plotx = np.append(plotx, np.mean(x[i * 4:i * 4 + 3]))
        plt.scatter(ploty[1:], plotx[1:])

plt.show()