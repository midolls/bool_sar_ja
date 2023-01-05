import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 8 # number of bits

df = pd.read_excel('RAMP_25S.xlsx',engine='openpyxl') #500번 simulation

df_list = df.values.tolist()


## Plot input
time = list(zip(*df_list))[0]
DO = list(zip(*df_list))[1]
IN = list(zip(*df_list))[5]
OE = list(zip(*df_list))[3]


time=list(time)
DO=list(map(int,DO))
IN =list(IN)
OE =list(OE)


fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.plot(time,DO,label='DO')
#ax1.plot(time,IN,label='IN')
ax1.legend()
ax1.grid(which='both',linestyle='dashed')

##Histogram
H=[0]*256
for i in range(len(time)):
    H[DO[i]]=H[DO[i]]+1

Hm_up = 0
for k in range(2**n-2):
    Hm_up = Hm_up+H[k]

Hm = Hm_up/(2**n-2)

##DLE
DLE=[0]*(2**n)

for i in range(2**n):
    DLE[i] = (H[i]-Hm)/Hm

##ILE
ILE=[0]*(2**n)
for i in range(2**n):
    for k in range(i):
        ILE[i]+=DLE[k]

fig2=plt.figure()
ax1=fig2.add_subplot(111)
ax1.plot(DLE,label='DLE')
ax1.plot(ILE,label='ILE')
ax1.legend()
ax1.grid(which='both',linestyle='dashed')
plt.show()


##리스트 나누기
#
# divide=np.where(abs(np.diff(DO))>50)[0]
# n_DOs=[]
# n_times=[]
# DO=np.array(DO)
# n_DOs.append(DO[0:divide[0]+1])
# n_times.append(time[0:divide[0]+1])
# for i in range(len(divide)-1):
#     n_DOs.append(DO[divide[i]+1:divide[i+1]+1])
#     n_times.append(time[divide[i]+1:divide[i+1]+1])
# Vins=[]
# for i in range(len(n_times)):
#     Vins.append(np.linspace(0,5,len(n_times[i])))
#

##plot
# fig2=plt.figure()
# Vin = np.linspace(0,5,len(n_times[i]))
# ax2=fig2.add_subplot(111)
# ax2.plot(Vin,(Vin)/Vlsb,color='purple')
#
#
# dif=[]
# for i in range(len(Vins)):
#     ax2.scatter(n_times[i],n_DOs[i],c='dodgerblue',s=10)
#     dif.append(np.where(abs(np.diff(n_DOs[i])==1))[0])
#
#
# #for i in range(len(dif)):
# #    ax2.scatter(n_times[i][dif[i]],n_DOs[i][dif[i]],c='violet',s=20)
#
# outs= [[0] * 1] * (2**N)
#
#
# for i in range(len(dif)):
#     for index,value in enumerate(dif[i]):
#         out=n_DOs[i][value]
#         outs[out].append(value)
#
#
#
# ax2.legend()
# ax2.grid(which='both',linestyle='dashed')
# plt.show()




