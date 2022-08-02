import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

n = 10 # number of bits

df = pd.read_excel('120u_10e.xlsx',engine='openpyxl')

df_list = df.values.tolist()

volt = list(zip(*df_list))[1]

### Resampling function
def resampling(vlt):
    vlt = [elem for elem in volt if abs(elem) > 0.5 or abs(elem) < 0.001]

    vlt_df=np.where(abs(np.diff(vlt))>0.499)[0]

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
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)



amplitude = resampling(volt)

t = np.arange(2,120,2)
amplitude=np.array(amplitude)


fig1,ax=plt.subplots(nrows=2,ncols=1,figsize=(8,6))

ax[0].stem(t, amplitude)
print(amplitude)
#ax[0].set_ylim(0,1)
#ax[0].legend()
ax[0].grid(True)

s_fft = np.fft.fft(amplitude-511.5, n=2*len(t)) # 추후 IFFT를 위해 abs를 취하지 않은 값을 저장한다.
amplitude2 = abs(s_fft)*(2/len(s_fft)) # 2/len(s)을 곱해줘서 원래의 amp를 구한다.
frequency = np.fft.fftfreq(len(s_fft),  2)

ax[1].stem(frequency, amplitude2)
#ax[1].legend()
ax[1].grid(True)


plt.show()

PS = 0
coef=np.where(amplitude2 > 60)[0]
for i in coef:
    PS += amplitude2[i]**2
    amplitude2[i] = 0


PDN = 0
for n in amplitude2:
    PDN += n**2

SNDR = 10*np.log10(PS/PDN)
ENOB = (SNDR - 1.76)/6.02



