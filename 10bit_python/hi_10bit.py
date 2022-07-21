import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## 1. parameters
Vref=1          # maximum voltage
N=10             # total bits
Ns=10000        # total signal samples
Vfs=2*Vref      # full scale
Vlsb=Vfs/2**N   # vlsb
SNR=30          # SNR in dB
Vcm=0.5         # common mode voltage of input voltage (0~1) v+v-

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
raw_data = {'Vin' : Vin,'Vp' : Vp,'Vn' : Vn} #리스트 자료형으로 생성
raw_data = pd.DataFrame(raw_data) #데이터 프레임으로 전환
raw_data.to_excel(excel_writer='Vin.xlsx') #엑셀로 저장

## 5. ideal SAR behavior and error with offset
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

ideal_off=(Vin-offset)/Vlsb+2**(N-1)
ideal_off=np.array(ideal_off)
err=(Vin-offset)/Vlsb+2**(N-1)-code2
err=np.array(err)
# plot error with offset
ax[1].plot(Vin,(Vin-offset)/Vlsb+2**(N-1)-code2)
ax[1].set_xlim(-1.2,1.2)
ax[1].grid()

ideal_off=(Vin-offset)/Vlsb+2**(N-1)
ideal_off=np.array(ideal_off)
err=(Vin-offset)/Vlsb+2**(N-1)-code2
err=np.array(err)

raw_data_i= {'Vin' :Vin,'ideal_off':ideal_off,'code_ideal':code2,'err':err}
raw_data_i = pd.DataFrame(raw_data_i) #데이터 프레임으로 전환
raw_data_i.to_excel(excel_writer='Videal_behavior.xlsx') #엑셀로 저장

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
ax[2].set_ylim(-0.5,0.5)
ax[2].legend()

plt.show()

raw_data_i_ID = {'INL' : INL2,'DNL' : DNL2}
raw_data_i_ID = pd.DataFrame(raw_data_i_ID)
raw_data_i_ID.to_excel(excel_writer='Videal_ID.xlsx')




##평균값 낸거
sigma=Vlsb/np.sqrt(12)/10**(SNR/20)
spark = []
rep_n = 10
for _ in range(rep_n):
    noise = np.random.normal(0,sigma,Ns)
    Vin=np.linspace(-1.2,1.2,Ns)+noise
    Vinn=np.linspace(-1.2,1.2,Ns)
    Vin_s=np.sort(Vin)
    Vp,Vn=balun(Vin)
    Vps,Vns=balun(Vin_s)

    code4=[]

    for idx in range(len(Vin)):
        code4.append(saradc(Vps[idx], Vns[idx]))
    code4=np.array(code4)

    offset=Vlsb/2


    spark.append(np.where(np.diff(code4)>0)[0])


sum_list = [0]*(2**(N-1))
avg_list = []
for i in range(rep_n):
    for j in range(2**(N-1)):
        sum_list[j] += spark[i][j]

for j in range(2**(N-1)):
    avg_list.append(int(sum_list[j]/rep_n))

print(avg_list)

fig5,ax=plt.subplots(nrows=3,ncols=1,figsize=(8,6))
# plot sar adc behavior with offset
ax[0].plot(Vin_s,code4,label='code with offset&noise',linestyle='-')
ax[0].plot(Vin,(Vin-offset)/Vlsb+2**(N-1),linestyle='--',label='ideal')
ax[0].set_yticks(np.arange(2**N))
ax[0].set_xlim(-1.2,1.2)
ax[0].legend(); ax[0].grid()

# plot error with offset
ax[1].plot(Vinn,(Vinn-offset)/Vlsb+2**(N-1)-code4)
ax[1].set_xlim(-1.2,1.2)
ax[1].grid()


ideal_off=(Vin-offset)/Vlsb+2**(N-1)
ideal_off=np.array(ideal_off)
err=(Vin-offset)/Vlsb+2**(N-1)-code2
err=np.array(err)

raw_data_n= {'Vin_noise' :Vin_s,'ideal_off':ideal_off,'code4':code4,'err':err}
raw_data_n = pd.DataFrame(raw_data_n) #데이터 프레임으로 전환
raw_data_n.to_excel(excel_writer='Vnoise_behavior.xlsx') #엑셀로 저장

# plot INL, DNL
inval_idx=np.where(np.logical_or(Vin<-1,Vin>1))[0]
code4[inval_idx]=-1     # not considering out of bound bits
code_idx=avg_list
#for idx in range(2**N-1):
#    arr=np.where(np.logical_or(code4==idx,code4==idx+1))
#    code_idx.append(int(np.mean(arr)))

for i in range(2**(N-1)):
    ax[0].scatter(Vin[code_idx[i]],code4[code_idx[i]],c='red',s=50)

diff=np.diff(code_idx)
DNL4=[0]; INL4=[0]
for idx in range(len(diff)):
    DNL4.append(diff[idx]/Ns/Vlsb*2.4-1)
    INL4.append((code_idx[idx+1]-code_idx[0])/Ns/Vlsb*2.4-idx-1)
DNL4.append(0); INL4.append(0)

ax[2].plot(DNL4,label='DNL')
ax[2].plot(INL4,label='INL')
ax[2].set_ylim(-0.5,0.5)
ax[2].legend()
#plt.show()

raw_data_n_ID = {'INL_Vnoise' : INL4,'DNL_Vnoise' : DNL4}
raw_data_n_ID = pd.DataFrame(raw_data_n_ID)
raw_data_n_ID.to_excel(excel_writer='Vnoise_ID.xlsx')

##ENOB without noise

def sin_wave(amp, freq, time):
    return amp * np.sin(2*np.pi*freq*time)

Ts=0.00001
Fs=1/Ts
Fn=Fs/2
f=10
T=1/f
Nsam=T/Ts

time = np.arange(0, T, Ts)
Vin = sin_wave(1, f, time)
Vp,Vn=balun(Vin)


sin_code1=[]
for idx in range(len(Vin)):
    sin_code1.append(saradc(Vp[idx], Vn[idx]))
sin_code1=np.array(sin_code1)

#sin_code1=sin_code1*Vlsb-1

fig6,ax=plt.subplots(nrows=2,ncols=1,figsize=(8,6))
# plot sar adc behavior without offset
ax[0].plot(time,Vin,label='Vin')
ax[0].plot(time,Vp,label='Vp')
ax[0].plot(time,Vn,label='Vn')
ax[0].set_xlim(0,T)
ax[0].grid(which='both',linestyle='dashed')
ax[0].legend(); ax[0].grid()

sin_code1=(sin_code1-511.5)

ax[1].plot(time,sin_code1,label='sin_code1')
ax[1].set_xlim(0,T)
ax[1].legend(); ax[1].grid()

signal = sin_code1

fig7,ax=plt.subplots(nrows=2,ncols=1,figsize=(8,6))
# s_fft = np.fft.fft(signal) # 추후 IFFT를 위해 abs를 취하지 않은 값을 저장한다.
s_fft = np.fft.fft(signal,int(2*Nsam)) # 추후 IFFT를 위해 abs를 취하지 않은 값을 저장한다.
amplitude = abs(s_fft)*(2/len(s_fft)) # 2/len(s)을 곱해줘서 원래의 amp를 구한다.
frequency = np.fft.fftfreq(len(s_fft), Ts)


coef=np.where(np.logical_and(amplitude > 0.01,abs(frequency)>25))[0]

PD_list = []
for i in range(len(coef)):
#    ax[0].scatter(frequency[coef[i]],amplitude[coef[i]],c='red',s=50)
    PD_list.append(amplitude[coef[i]]**2)

PN_coef=np.where(amplitude < 25.5)[0]
PN_list = []
for i in range(len(PN_coef)):
#    #ax[0].scatter(frequency[PN_coef[i]],amplitude[PN_coef[i]],c='green',s=50)
    PN_list.append(amplitude[PN_coef[i]]**2)

PD = sum(PD_list)
PD=0
#PN = (Vlsb/np.sqrt(12))**2
PN = sum(PN_list)



PS_list = []
sig_coef=np.where(amplitude>25.5)[0]
for i in range(len(sig_coef)):
    ax[0].scatter(frequency[sig_coef[i]],amplitude[sig_coef[i]],c='green',s=50)
    PS_list.append(amplitude[sig_coef[i]]**2)

PS = sum(PS_list)

SNDR = 10*np.log10(PS/(PN + PD))
ENOB = (SNDR - 1.76)/6.02


ax[0].stem(frequency, amplitude,label='fft')
ax[0].set_xlim(-Fn,Fn)
#ax[0].set_ylim(0,0.5)
ax[0].legend()
ax[0].grid(True)

#plt.show()

raw_data_sin = {'Vin' : Vin,'sin_code1' :sin_code1}
raw_data_fft = {'frequency' : frequency,'amplitude' :amplitude}
#raw_data_noise_eID = {'INL_Vnoise' : INL4,'DNL_Vnoise' : DNL4}
raw_data_sin = pd.DataFrame(raw_data_sin) #데이터 프레임으로 전환
raw_data_fft = pd.DataFrame(raw_data_fft) #데이터 프레임으로 전환
#raw_data_noise_eID = pd.DataFrame(raw_data_noise_eID)
raw_data_sin.to_excel(excel_writer='sin.xlsx') #엑셀로 저장
raw_data_fft.to_excel(excel_writer='sinfft.xlsx') #엑셀로 저장
#raw_data_noise_eID.to_excel(excel_writer='Vnoise_ID.xlsx')


### input FFT

#plt.show()

##ENOB with noise(fig 8~9)
def sin_wave(amp, freq, time):
    return amp * np.sin(2*np.pi*freq*time)

SNR=20
#20log(255/25.5)=20
noise = np.random.normal(0,Vlsb/np.sqrt(12)/10**(SNR/20),10000)
#freq = 1

time = np.arange(0, T, Ts)
Vin = sin_wave(1, f, time)


sin_code2=[]

for idx in range(len(Vin)):
    Vin[idx]=Vin[idx]+noise[idx]



Vp,Vn=balun(Vin)

for idx in range(len(Vin)):
    sin_code2.append(saradc(Vp[idx], Vn[idx]))
sin_code2=np.array(sin_code2)

#sin_code1=sin_code1*Vlsb-1

fig8,ax=plt.subplots(nrows=2,ncols=1,figsize=(8,6))
# plot sar adc behavior without offset
ax[0].plot(time,Vin,label='Vin')
ax[0].plot(time,Vp,label='Vp')
ax[0].plot(time,Vn,label='Vn')
ax[0].set_xlim(0,0.1)
ax[0].grid(which='both',linestyle='dashed')
ax[0].legend(); ax[0].grid()

sin_code2=(sin_code2-511.5)

ax[1].plot(time,sin_code2,label='sin_code2')
ax[1].set_xlim(0,0.1)
ax[1].legend(); ax[1].grid()

signal = sin_code2

fig9,ax=plt.subplots(nrows=2,ncols=1,figsize=(8,6))
# s_fft = np.fft.fft(signal) # 추후 IFFT를 위해 abs를 취하지 않은 값을 저장한다.
s_fft = np.fft.fft(signal,n=int(2*Nsam)) # 추후 IFFT를 위해 abs를 취하지 않은 값을 저장한다.
amplitude = abs(s_fft)*(2/len(s_fft)) # 2/len(s)을 곱해줘서 원래의 amp를 구한다.
frequency = np.fft.fftfreq(len(s_fft),  Ts)

coef=np.where(np.logical_and(amplitude > 0.05,abs(frequency)>25))[0]

PD_list = []
for i in range(len(coef)):
#    ax[0].scatter(frequency[coef[i]],amplitude[coef[i]],c='red',s=50)
    PD_list.append(amplitude[coef[i]]**2)

PN_coef=np.where(amplitude < 25.5)[0]
PN_list = []
for i in range(len(PN_coef)):
#    #ax[0].scatter(frequency[PN_coef[i]],amplitude[PN_coef[i]],c='green',s=50)
    PN_list.append(amplitude[PN_coef[i]]**2)

PD = sum(PD_list)
PD=0
#PN = (Vlsb/np.sqrt(12))**2
PN = sum(PN_list)



PS_list = []
sig_coef=np.where(amplitude>25.5)[0]
for i in range(len(sig_coef)):
    ax[0].scatter(frequency[sig_coef[i]],amplitude[sig_coef[i]],c='green',s=50)
    PS_list.append(amplitude[sig_coef[i]]**2)

PS = sum(PS_list)
#PS = 2*amplitude[sig_coef[0]]**2
#PS = 0.87**2/2



SNDR = 10*np.log10(PS/(PN + PD))
N_ENOB = (SNDR - 1.76)/6.02


ax[0].stem(frequency, amplitude,label='fft')
ax[0].set_xlim(-Fn,Fn)
#ax[0].set_ylim(0,0.5)
ax[0].legend()
ax[0].grid(True)

#plt.show()

raw_data_nsin = {'Vin' : Vin,'sin_code2' :sin_code2}
raw_data_nfft = {'frequency' : frequency,'amplitude' :amplitude}
#raw_data_noise_eID = {'INL_Vnoise' : INL4,'DNL_Vnoise' : DNL4}
raw_data_nsin = pd.DataFrame(raw_data_nsin) #데이터 프레임으로 전환
raw_data_nfft = pd.DataFrame(raw_data_nfft) #데이터 프레임으로 전환
#raw_data_noise_eID = pd.DataFrame(raw_data_noise_eID)
raw_data_nsin.to_excel(excel_writer='nsin.xlsx') #엑셀로 저장
raw_data_nfft.to_excel(excel_writer='nsinfft.xlsx') #엑셀로 저장

### input FFT



'''








