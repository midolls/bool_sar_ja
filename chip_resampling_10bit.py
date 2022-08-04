import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

n = 8 # number of bits

df = pd.read_excel('CHIPTEST_sin.xlsx',engine='openpyxl')

df_list = df.values.tolist()

## Plot input
time = list(zip(*df_list))[0]
clk=list(zip(*df_list))[3]
sar = list(zip(*df_list))[4]
DO = list(zip(*df_list))[5]

time=np.array(time)
clk=np.array(clk)
sar=np.array(sar)
DO=np.array(DO)


fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.plot(time,DO,label='DO')
ax1.plot(time,clk,label='clk')
ax1.plot(time,sar,label='sar')
ax1.legend()
ax1.grid(which='both',linestyle='dashed')

## Resampling

start = time[0]

fe=np.where(np.diff(clk)==-1)[0]
out=np.where(sar==1)[0]
samplingtime=[]

for i in fe:
    for j in out:
        if i==j:
            samplingtime.append(i+2)


for i in range(len(samplingtime)):
        ax1.scatter(time[samplingtime[i]],0, c='red',s=30)


plt.show()

number = np.where(np.diff(samplingtime)>70)[0]
deci = []
sample = []
for i in number:
    sample = DO[samplingtime[i-7:i+1]]
    sample = sample.astype(int)
    deci.append(int(''.join(str(_) for _ in sample),2))

fig2=plt.figure()
ax2=fig2.add_subplot(111)
ax2.plot(deci,label='DO')
ax2.legend()
ax2.grid(which='both',linestyle='dashed')
plt.show()

raw_data = {'sin_chip' : deci} #리스트 자료형으로 생성
raw_data = pd.DataFrame(raw_data) #데이터 프레임으로 전환
raw_data.to_excel(excel_writer='sin_chip.xlsx') #엑셀로 저장

##INL DNL

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



