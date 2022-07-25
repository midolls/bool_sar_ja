import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## 1. parameters
Vref=1          # maximum voltage
N=3             # total bits
Ns=10000        # total signal samples
Vfs=2*Vref      # full scale
Vlsb=Vfs/2**N   # vlsb
SNR=30          # SNR in dB
Vcm=0.5         # common mode voltage of input
print('Vlsb =',Vlsb); print('Vfs =',Vfs)

## 2. saradc behavior function

#  SAR ADC: vip, vin --> digital code

def saradc(vip,vin,N=N,Vref=Vref,show=False):
    '''
    Parameters:
        vip: differential input voltage +
        vin: differential input voltage -
        N: number of bits
        Vref: reference voltage
        show: show capacitor network and its voltage
    '''
    B=''
    vp=vip; vn=vin
    for i in range(N):
        if vp>=vn:
            B+='1'
            vp -= Vref/2**(i+1)
        else:
            B+='0'
            vn -= Vref/2**(i+1)

    # print out capacitor network and its voltage
    if show==True:
        spacing=9; middle=spacing//2+1
        # voltage at cap
        vpcap=''; vncap=''
        for bit in B[:-1]:
            if bit=='1':
                vpcap+='Vref'+' '*(spacing-4)
                vncap+='gnd'+' '*(spacing-3)
            else:
                vpcap+='gnd'+' '*(spacing-3)
                vncap+='Vref'+' '*(spacing-4)
        # last capacitor's voltage doesn't change
        vpcap+='Vref'+' '*(spacing-4)
        vncap+='Vref'+' '*(spacing-4)

        line='|'+' '*(spacing-1)
        cap='-'+' '*(spacing-1)

        # show
        print()
        print(vpcap)
        print(' '+line*N)
        print(' '+cap*N); print(' '+cap*N)
        print(' '+line*N+'|\\')
        print('-'*spacing*N+' |+\\')
        print('-'*spacing*N+' |-/')
        print(' '+line*N+'|/')
        print(' '+cap*N); print(' '+cap*N)
        print(' '+line*N)
        print(vncap)
        print('output: '+B)
        print()
    return int(B,2)

# test adc function
print(saradc(0.3,0,show=True))
print(saradc(0,0.5,show=True))

## 3. differential input voltage

# input voltage
Vin=np.linspace(-1.2,1.2,Ns)
print(Vin)

# single ended input to differential input
def balun(Vin):
    '''
    Parameter:
        Vin:
    '''
    Vp=Vcm+Vin/2
    Vn=Vcm-Vin/2
    return (Vp,Vn)
Vp,Vn=balun(Vin)
print(Vp); print(Vn)

# plot Vin, Vp, Vn
fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.plot(Vin,Vin,label='Vin')
ax1.plot(Vin,Vp,label='Vp')
ax1.plot(Vin,Vn,label='Vn')
ax1.legend()
ax1.set_xticks(np.arange(-1.2,1.3,0.2))
ax1.set_yticks(np.arange(-1.2,1.3,0.2))
ax1.grid(which='both',linestyle='dashed')
plt.show()


## ideal SAR behavior and error with offset
offset=Vlsb/2
code2=[]
for idx in range(len(Vin)):
    code2.append(saradc(Vp[idx],Vn[idx]))
code2=np.array(code2)

fig3,ax=plt.subplots(nrows=3,ncols=1,figsize=(8,6))
# plot sar adc behavior with offset
ax[0].plot(Vin,code2,label='code with offset')
ax[0].plot(Vin,(Vin-offset)/Vlsb+2**(N-1),linestyle='--',label='ideal')
ax[0].set_yticks(np.arange(2**N))
ax[0].set_xlim(-1.2,1.2)
ax[0].legend(); ax[0].grid()

# plot error with offset
ax[1].plot(Vin,(Vin-offset)/Vlsb+2**(N-1)-code2)
ax[1].set_xlim(-1.2,1.2)
ax[1].grid()

# plot INL, DNL
inval_idx=np.where(np.logical_or(Vin<-1,Vin>1))[0]
code2[inval_idx]=-1     # not considering out of bound bits
code_idx=[]
for idx in range(2**N-1):
    arr=np.where(np.logical_or(code2==idx,code2==idx+1))
    code_idx.append(int(np.mean(arr)))

diff=np.diff(code_idx)
DNL2=[0]; INL2=[0]
for idx in range(len(diff)):
    DNL2.append(diff[idx]/Ns/Vlsb*2.4-1)
    INL2.append((code_idx[idx+1]-code_idx[0])/Ns/Vlsb*2.4-idx-1)
DNL2.append(0); INL2.append(0)

ax[2].plot(DNL2,label='DNL')
ax[2].plot(INL2,label='INL')
ax[2].set_ylim(-0.01,0.01)
ax[2].legend()
plt.show()


##Ramp with noise
sigma=Vlsb/np.sqrt(12)/10**(SNR/20)
spark = []
rep_n = 50
Vinn=np.linspace(-1.2,1.2,Ns)
Vp,Vn=balun(Vin)
offset=Vlsb/2

#input with noise

for _ in range(rep_n):
    noise = np.random.normal(0,sigma,Ns)
    Vin=np.linspace(-1.2,1.2,Ns)+noise
    Vin_s=np.sort(Vin)
    Vps,Vns=balun(Vin_s)

    code4=[]

    for idx in range(len(Vin)):
        code4.append(saradc(Vps[idx], Vns[idx]))
    code4=np.array(code4)
    spark.append(np.where(np.diff(code4)>0)[0])

#Find index using mean value

sum_list = [0]*(2**N-1)
avg_list = []
for i in range(rep_n):
    for j in range(2**N-1):
        sum_list[j] += spark[i][j]

for j in range(2**N-1):
    avg_list.append(int(sum_list[j]/rep_n))

# plot sar adc behavior with offset
fig5,ax=plt.subplots(nrows=3,ncols=1,figsize=(8,6))
ax[0].plot(Vin_s,code4,label='code with offset&noise',linestyle='-')
ax[0].plot(Vin,(Vin-offset)/Vlsb+2**(N-1),linestyle='--',label='ideal')
ax[0].set_yticks(np.arange(2**N))
ax[0].set_xlim(-1.2,1.2)
ax[0].legend(); ax[0].grid()

# plot error with offset
ax[1].plot(Vinn,(Vinn-offset)/Vlsb+2**(N-1)-code4)
ax[1].set_xlim(-1.2,1.2)
ax[1].grid()

# plot INL, DNL
inval_idx=np.where(np.logical_or(Vin<-1,Vin>1))[0]
code4[inval_idx]=-1     # not considering out of bound bits
code_idx=avg_list

for i in range(2**N-1):
    ax[0].scatter(Vin[code_idx[i]],code4[code_idx[i]],c='red',s=50)

diff=np.diff(code_idx)
DNL4=[0]; INL4=[0]
for idx in range(len(diff)):
    DNL4.append(diff[idx]/Ns/Vlsb*2.4-1)
    INL4.append((code_idx[idx+1]-code_idx[0])/Ns/Vlsb*2.4-idx-1)
DNL4.append(0); INL4.append(0)

ax[2].plot(DNL4,label='DNL')
ax[2].plot(INL4,label='INL')
ax[2].set_ylim(-0.005,0.005)
ax[2].legend()

plt.show()

##ENOB without noise

def sin_wave(amp, freq, time):
    return amp * np.sin(2*np.pi*freq*time)

#Initilization
Ts=0.001
Fs=1/Ts
Fn=Fs/2
f=10
T=1/f
Nsam=T/Ts

time = np.arange(0, 1, Ts)
Vin = sin_wave(1, f, time)
Vp,Vn=balun(Vin)

#ADC
sin_code1=[]
for idx in range(len(Vin)):
    sin_code1.append(saradc(Vp[idx], Vn[idx]))
sin_code1=np.array(sin_code1)

#plot
fig6,ax=plt.subplots(nrows=2,ncols=1,figsize=(8,6))

ax[0].plot(time,Vin,label='Vin')
ax[0].plot(time,Vp,label='Vp')
ax[0].plot(time,Vn,label='Vn')
ax[0].set_xlim(0,1)
ax[0].grid(which='both',linestyle='dashed')
ax[0].legend(); ax[0].grid()

ax[1].plot(time,sin_code1,label='sin_code1')
ax[1].set_xlim(0,1)
ax[1].legend(); ax[1].grid()

#fft
sin_code1=sin_code1-3.5 #DC값 제거
signal = sin_code1
s_fft = np.fft.fft(signal,n=int(100*Nsam))
amplitude = abs(s_fft)*(100/len(s_fft))
frequency = np.fft.fftfreq(len(s_fft),  Ts)

PN_coef=np.where(abs(frequency)>20)[0]
PN_list = []
for i in range(len(PN_coef)):
    PN_list.append(amplitude[PN_coef[i]]**2)

PD = 0
PN = sum(PN_list)


PS_list = []
sig_coef=np.where(abs(frequency)<20)[0]
for i in range(len(sig_coef)):
    PS_list.append(amplitude[sig_coef[i]]**2)

PS = sum(PS_list)


SNDR = 10*np.log10(PS/(PN + PD))
ENOB = (SNDR - 1.76)/6.02

fig7=plt.figure()
ax7=fig7.add_subplot(111)

ax7.stem(frequency, amplitude,label='fft(10Hz)')
ax7.set_xlim(-Fn,Fn)
ax7.legend()
ax7.grid(True)

raw_data = {'10Hz_ideal' : signal} #리스트 자료형으로 생성
raw_data = pd.DataFrame(raw_data) #데이터 프레임으로 전환
raw_data.to_excel(excel_writer='10Hz_ideal.xlsx') #엑셀로 저장

##ENOB with F=10 with noise

def sin_wave(amp, freq, time):
    return amp * np.sin(2*np.pi*freq*time)

#INITIalization
SNR=20
noise = np.random.normal(0,Vlsb/np.sqrt(12)/10**(SNR/20),2000)

Ts=0.001
Fs=1/Ts
Fn=Fs/2
f=1
T=1/f
Nsam=T/Ts

time = np.arange(0, T, Ts)
Vin = sin_wave(1, f, time)


sin_code2=[]

for idx in range(len(Vin)):
    Vin[idx]=Vin[idx]+noise[idx]



Vp,Vn=balun(Vin)

#SAR ADC
for idx in range(len(Vin)):
    sin_code2.append(saradc(Vp[idx], Vn[idx]))
sin_code2=np.array(sin_code2)


#plot Vin
fig8,ax=plt.subplots(nrows=2,ncols=1,figsize=(8,6))

ax[0].plot(time,Vin,label='Vin')
ax[0].plot(time,Vp,label='Vp')
ax[0].plot(time,Vn,label='Vn')
ax[0].set_xlim(0,T)
ax[0].grid(which='both',linestyle='dashed')
ax[0].legend(); ax[0].grid()

ax[1].plot(time,sin_code2,label='sin_code2')
ax[1].set_xlim(0,T)
ax[1].legend(); ax[1].grid()

#FFT
sin_code2=sin_code2-3.5
signal = sin_code2


s_fft = np.fft.fft(signal,n=int(1000*Nsam))
amplitude = abs(s_fft)*(1000/len(s_fft))
frequency = np.fft.fftfreq(len(s_fft),  Ts)

#noise in FFT
PN_coef=np.where(abs(frequency)>2*f)[0]
PN_list = []
for i in range(len(PN_coef)):
    PN_list.append(amplitude[PN_coef[i]]**2)
PN = sum(PN_list)

#signal in FFT
PS_list = []
sig_coef=np.where(abs(frequency)<2*f)[0]
for i in range(len(sig_coef)):
    PS_list.append(amplitude[sig_coef[i]]**2)
PS = sum(PS_list)


SNDR = 10*np.log10(PS/(PN + PD))
N_ENOB = (SNDR - 1.76)/6.02

fig9=plt.figure()
ax9=fig9.add_subplot(111)
ax9.stem(frequency, amplitude,label='fft(with noise)')
ax9.set_xlim(-Fn,Fn)
ax9.legend()
ax9.grid(True)


raw_data = {'1Hz_noise' : signal} #리스트 자료형으로 생성
raw_data = pd.DataFrame(raw_data) #데이터 프레임으로 전환
raw_data.to_excel(excel_writer='1Hz_noise.xlsx') #엑셀로 저장









