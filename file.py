import numpy as np
import scipy as sp
import scipy.fftpack
import matplotlib.pyplot as plt

# Aplicação da FFT no sinal

def spectrum(y,Fs):
	n = len(y) # length of the signal
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	frq = frq[range(n//2)] # one side frequency range
	Y = sp.fftpack.fft(y)/n # fft computing and normalization
	Y = Y[range(n//2)] 
	return frq, abs(Y)

freq = 100 #freq do sinal
Fs = 4*freq  #freq de amostragem
Ts = 1/Fs #intervalo de amostragem

t = np.arange(0, 5, Ts) #comprimento de amostra
ys = np.sin(2*np.pi*freq*t) #sinal

f,a = spectrum(ys,Fs)

#plt.plot(f, a)
#plt.show()

# Identificação da nota

i = np.where(a == max(a))
mf = np.asscalar(f[i])
print('%.4f Hz' % mf)

mapa = np.fromfunction(lambda i, j: (2)**(j-3) * 440*(2**(1/12))**(i-9), (12, 7))# Equanção para mapeamento da nota na escala musical
notas = ['C','C#/Db','D','D#/Eb','E','F','F#/Gb','G','G#/Ab','A','A#/Bb','B']

if not mapa[0,0] < mf < mapa[-1,-1]:
    print('Não há tal nota')
    exit(0)

i,j = np.array(np.where(mapa.flat[np.abs(mapa - mf).argmin()] == mapa)).flatten()

nota = notas[i]
altura = j+1

print('nota: %s%s'%(nota,altura))
