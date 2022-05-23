import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
from scipy.io.wavfile import write

# Load data
data, samplerate = sf.read(r'C:\Users\Nikolai Lund Kühne\OneDrive - Aalborg Universitet\Uni\4. semester\P4 - Signaler og Systemer\ingridstoej.wav')


def downsample(array,n,phase):
    '''
    Parameters
    ----------
    array : Array
        Array with data to be downsampled.
    n : Integer
        Downsampling factor.
    phase : Integer
        Phase shift of the downsampling.
    Returns
    -------
    Array
        Downsampled array.
    '''
    return array[phase::n]

Fs = 14700 # Samplefrequency 44.1kHz/3 = 14.7Hz

downsampled = downsample(data,3,2) # Downsampled Audio

# Define the different compression methods and the Hearing Loss Simulator (HLS)

def lincompression(lyd, a):
    '''
    Parameters
    ----------
    lyd : Array
        Array of sound signal.
    a : Float
        Compression factor.

    Returns
    -------
    x : Array
        Compressed sound signal.
    '''
    f, t, Zxx = signal.stft(lyd, fs = Fs, window = 'hann', nperseg=512)
    arr = np.zeros((len(Zxx), len(Zxx[0])),dtype = 'complex_')
    for i in range(len(Zxx[0])):
        for j in range(len(Zxx)):   
            b = round(a*j)
            arr[b,i] += Zxx[j,i]
    t_ny, x = signal.istft(arr,fs = Fs, window = 'hann', nperseg=512)
    return x


def piecelincompression(lyd, a, cutoff):
    '''
    Parameters
    ----------
    lyd : Array
        Array of sound signal.
    a : Array
        Array of 3 chosen compression factors.
    cutoff : Array
        Array of 2 chosen corner frequencies.

    Returns
    -------
    x : Array
        Piecewise compressed sound signal.
    '''
    f, t, Zxx = signal.stft(lyd, fs = Fs, window = 'hann', nperseg=512)
    arr1 = np.zeros((len(Zxx), len(Zxx[0])),dtype = 'complex_')
    for i in range(len(Zxx[0])):
        for j in range(len(Zxx)):  
            if f[j] <= cutoff[0]:
                b = round(a[0]*j)
                # print(b) #slut 17
                arr1[b,i] += Zxx[j,i]
                j1 = j
            elif f[j] <= cutoff[1]:
                b = round(a[1]*j + (a[0]*j1 - a[1]*j1))
                # print(b) #slut=34
                arr1[b,i] += Zxx[j,i]
                j2 = j
            else:
                b = round(a[2]*j + (j1 - a[1]*j1 + a[1]*j2 - a[2]*j2))
                # print(b) #start=74 flyttes til start=34
                arr1[b,i] += Zxx[j,i]
    t_ny, x = signal.istft(arr1,fs = Fs, window = 'hann', nperseg=512)
    return x


def transposition(lyd, cutoff, tau):
    '''
    Parameters
    ----------
    lyd : Array
        Array of sound signal.
    cutoff : Integer or Float
        Corner frequency.
    tau : Integer or Float
        Transposition shifting factor.

    Returns
    -------
    x : Array
        Frequency transposed sound signal.
    '''
    f, t, Zxx = signal.stft(lyd, fs = Fs, window = 'hann', nperseg=512)
    arr1 = np.zeros((len(Zxx), len(Zxx[0])),dtype = 'complex_')
    arr2 = np.zeros((len(Zxx), len(Zxx[0])),dtype = 'complex_')
    num_f = min(f, key=lambda x:abs(x-tau))
    for k in range(len(f)):
        if f[k] == num_f:
            tau_f = k
            # print(f'index = {b}')
            # print(f'index værdi = {f[b]}')
    for i in range(len(Zxx[0])):
        for j in range(len(Zxx)):   
            if f[j] >= cutoff:
                b = round(j - tau_f)
                arr1[b,i] += Zxx[j,i]
    for i in range(len(Zxx[0])):
        for j in range(len(Zxx)):
            arr2[j,i] = arr1[j,i] + Zxx[j,i]
    # print(len(array))
    # print(len(array[0]))
    t_ny, x = signal.istft(arr2, fs = Fs, window = 'hann', nperseg=512)
    return x


def transposition_comp(lyd, cutoff, tau, a):
    '''
    Parameters
    ----------
    lyd : Array
        Array of sound signal..
    cutoff : Integer or Float
        Corner frequency.
    tau : Integer or Float
        Transposition shifting factor, must be negative.
    a : Float
        Compression factor.

    Returns
    -------
    x : Array
        Frequency compressed and transposed sound signal.
    '''
    f, t, Zxx = signal.stft(lyd, fs = Fs, window = 'hann', nperseg=512)
    arr = np.zeros((len(Zxx), len(Zxx[0])),dtype = 'complex_')
    array = np.zeros((len(Zxx), len(Zxx[0])),dtype = 'complex_')
    num_f = min(f, key=lambda x:abs(x-tau))
    for k in range(len(f)):
        if f[k] == num_f:
            tau_f = k
            # print(f'index = {b}')
            # print(f'index værdi = {f[b]}')
    for i in range(len(Zxx[0])):
        for j in range(len(Zxx)):   
            if f[j] >= cutoff:
                b = round((j - tau_f) * a)
                arr[b,i] += Zxx[j,i]
    for i in range(len(Zxx[0])):
        for j in range(len(Zxx)):
            array[j,i] = arr[j,i] + Zxx[j,i]
    t_ny, x = signal.istft(array,fs = Fs, window = 'hann', nperseg=512)
    return x

# compressedpiece = piecelincompression(data,a=[1,0.2,0.7],cutoff=[500, 3000])
# compressedlin = lincompression(downsampled,0.7)
# compressedtrans = transposition(downsampled, 2000, 1500)
# compressedboth = transposition_comp(downsampled, cutoff = 2000, tau = 1000, a = 0.7)

# With noise:
# compressedpiece = piecelincompression(data,a=[1,0.2,0.7],cutoff=[500, 3000])
# compressedlin = lincompression(data,0.7)
# compressedtrans = transposition(data, 2000, 1500)
# compressedboth = transposition_comp(data, cutoff = 2000, tau = 1000, a = 0.7)

# IIR HLS: comment out when using FIR HLS

def HLS(lyd):
    '''
    Parameters
    ----------
    lyd : Array
        Array of the sound signal.

    Returns
    -------
    y : Array
        Array of sound signal processed through the Hearing Loss Simulator.
    '''
    y = np.zeros(len(lyd))
    y[0] = 0.003062383609*lyd[0]
    y[1] = 0.003062383609*lyd[1]+0.009187150829*lyd[0]+2.361425149*y[0]
    y[2] = 0.003062383609*lyd[2]+0.009187150829*lyd[1]+0.009187150829*lyd[0]+2.361425149*y[1]-1.911136410*y[0]
    y[3] = 0.003062383609*lyd[3]+0.009187150829*lyd[2]+0.009187150829*lyd[1]+0.003062383609*lyd[0]+2.361425149*y[2]-1.911136410*y[1]+0.5252121917*y[0]
    for n in range(4,len(lyd)):
        y[n] = 0.003062383609*lyd[n] + 0.009187150829*lyd[n-1] + 0.009187150829*lyd[n-2] + 0.003062383609*lyd[n-3] + 2.361425149*y[n-1] - 1.911136410*y[n-2] + 0.5252121917*y[n-3]
    return y

# hearlosspiece = HLS(compressedpiece)
# hearlosstrans = HLS(compressedtrans)
# hearlossboth = HLS(compressedboth)



# FIR HLS: comment out when using IIR HLS
################ window method ###################
M1 = 128
M2 = 256
M3 = 512
M4 = 1024

fs = 14700
# fs*x = fc in Hz
fc = 0.102041*np.pi
    
alph1 = (M1-1)/2
alph2 = (M2-1)/2
alph3 = (M3-1)/2
alph4 = (M4-1)/2

n_1 = np.arange(M1)
n_2 = np.arange(M2)
n_3 = np.arange(M3)
n_4 = np.arange(M4)

v1 = signal.hamming(M1) # Hamming window
v2 = signal.hamming(M2) # Hamming window
v3 = signal.hamming(M3) # Hamming window
v4 = signal.hamming(M4) # Hamming window
   
# ideal impuls response
hd1 = np.sin(fc*(n_1-alph1))/(np.pi*(n_1-alph1))
hd2 = np.sin(fc*(n_2-alph2))/(np.pi*(n_2-alph2))
hd3 = np.sin(fc*(n_3-alph3))/(np.pi*(n_3-alph3))
hd4 = np.sin(fc*(n_4-alph4))/(np.pi*(n_4-alph4))

NaNs1 = np.isnan(hd1)
NaNs2 = np.isnan(hd2)
NaNs3 = np.isnan(hd3)
NaNs4 = np.isnan(hd4)

hd1[NaNs1] = 0.102041
hd2[NaNs2] = 0.102041
hd3[NaNs3] = 0.102041
hd4[NaNs4] = 0.102041
 
# reel impuls og amplitude respons for reel impuls
hv_1= hd1*v1
hv_2= hd2*v2
hv_3= hd3*v3
hv_4= hd4*v4

Hv1 = np.abs(fft.fft(hv_1, 10000)[:5000])
Hv2 = np.abs(fft.fft(hv_2, 10000)[:5000])
Hv3 = np.abs(fft.fft(hv_3, 10000)[:5000])
Hv4 = np.abs(fft.fft(hv_4, 10000)[:5000])

omegaH1 = np.linspace(0, np.pi, len(Hv1))
omegaH2 = np.linspace(0, np.pi, len(Hv2))
omegaH3 = np.linspace(0, np.pi, len(Hv3))
omegaH4 = np.linspace(0, np.pi, len(Hv4))

plt.figure(figsize = (10,6),dpi = 120)

# plot amplitude response:

# plt.plot(omegaH1, Hv1, 'green', label = f'Hamming window, filter order = {M1}')
# plt.plot(omegaH2, Hv2, 'r', label = f'Hamming window, filter order = {M2}')
# plt.plot(omegaH3, Hv3, 'b', label = f'Hamming window, filter order = {M3}')
# plt.plot(omegaH4, Hv4, 'black', label = f'Hamming window, filter order = {M4}')
# plt.title(r'Plot of the amplitude response $|H(\mathrm{e}^{j\omega})|$ for different filter orders')
# plt.xlabel("Frequency", fontsize = 12)
# plt.ylabel("Amplitude", fontsize = 12)
# plt.text(3.12,-0.09,r'$\pi$', fontsize = 12)
# plt.xlim(0,0.31)
# plt.ylim(0.98,1.02)

# Plot Gain instead
# a = [2*np.pi*3675/fs]
# b = [-85]
# plt.xlim(0,3.24)
# plt.plot(omegaH1, 20*np.log10(abs(Hv1)), 'green', label = f'Hamming window, filter order = {M1}')
# plt.plot(omegaH2, 20*np.log10(abs(Hv2)), 'r', label = f'Hamming window, filter order = {M2}')
# plt.plot(omegaH3, 20*np.log10(abs(Hv3)), 'b', label = f'Hamming window, filter order = {M3}')
plt.plot(omegaH4, 20*np.log10(abs(Hv4)), 'black', label = f'Hamming window, filter order = {M4}')
plt.title(r'Plot of the amplitude response $20\cdot log10(|H(\mathrm{e}^{j\omega})|)$ in dB')
plt.xlabel("Frequency", fontsize = 12)
plt.ylabel("Amplitude [dB]", fontsize = 12)
plt.text(3.12,-157,r'$\pi$', fontsize = 12)

plt.scatter(2*np.pi*3675/fs,-85, marker='x', c = ['r'])
    
plt.xlim(0,3.24)
plt.ylim(-150,7)
plt.vlines(2*np.pi*3675/fs, ymin = -150, ymax = -85, linestyles = 'dashed', color = 'r')
plt.hlines(-85, xmin = 0, xmax = 2*np.pi*3675/fs, linestyles = 'dashed', color = 'r')
plt.text(0.002,-91,'-85dB')
plt.rc('ytick', labelsize=12)   
plt.rc('xtick', labelsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid()
plt.show()
    


# Output is given by the convolution between the input and the impulseresponse of the FIR filter
result = np.convolve(data,hd4) #Always hd4 since the order is 1024
# time = np.linspace(0,4,32000)
# plt.specgram(result,Fs=14700, noverlap=128, scale = 'dB', cmap='inferno', window = np.hanning(256))
# cbar = plt.colorbar()
# cbar.set_label('Effekt/frekvens (dB/Hz)')
# cbar.minorticks_on()
# plt.ylabel('Frekvens [Hz]')
# plt.title('Spektrogram af støjreduceret talesignal')
# plt.xlabel('Tid [sec]')
# plt.show()
# Write wav-file
# write(r'C:\Users\Nikolai Lund Kühne\OneDrive - Aalborg Universitet\Uni\4. semester\P4 - Signaler og Systemer\ingridnoisyrevisedHLS.wav', Fs, result.astype(np.float32))


