import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

n = 3 # number of bits

df = pd.read_excel('10ue.xlsx',engine='openpyxl')

df_list = df.values.tolist()

volt = list(zip(*df_list))[1]

### Resampling function
def resampling(vlt):
    vlt = [elem for elem in volt if abs(elem) > 0.9 or abs(elem) < 0.001]

    vlt_df=np.where(abs(np.diff(vlt))>0.8)[0]

    deci = []
    sample = []
    cnt = 0
    for i in range(1,len(vlt_df),2):
        if vlt[vlt_df[i]] > 0:
            sample.append('1')
        elif vlt[vlt_df[i]] < 0:
            sample.append('0')
        cnt += 1
        if cnt == n:
            deci.append(int(''.join(sample),2))
            sample = []
            cnt = 0

    return deci

###
amplitude = resampling(volt)

t = np.arange(0.3,10,0.3)
amplitude=np.array(amplitude)

plt.stem(t,amplitude-3.5)

plt.show()

#vt = np.logical_or(volt>0.8,volt<0.1)
#idx=np.where(np.logical_or(volt>0.8,volt<0.1))[0]
