#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:32:03 2023

@author: ranshuai
"""
# BPSK
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
from scipy import signal
def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.bool)
# import 24 bit digital data
id_num = 2633609
Nbits = 24
tx_bin = bin_array(id_num, Nbits) # the 24 bit binarry array 
# When transmitting the last signal, errors always occur, 
# so I add several zero values to the original signal, making the last signal the third last signal.
tx_bin = np.append(tx_bin,0)
tx_bin = np.append(tx_bin,0)
Nbits = Nbits + 2
##############################################
bit_len = 16 #
fc = 0.125 # normalized frequency
s = np.copy(tx_bin) # s = original signal
print(tx_bin)
tx_mod = np.empty(0) # The modulated signal
plt.figure()
plt.plot(tx_bin)
plt.title("Original signal")
plt.xlim(0,24)
plt.show()

############### modulation###############
for i in range(0,Nbits):
    for j in range(bit_len):
        tx_mod = np.append(tx_mod,(2*s[i]-1)*np.cos(2*np.pi*fc*(i*bit_len+j)))# signal * coswct carrier frequency

plt.figure()
plt.plot(tx_mod)
plt.title("The modulated Signal")
plt.show()

plt.figure()
plt.plot(np.abs(fft.fft(tx_mod))) # converting a complex number to its magnitude.
plt.title("The Spectrum of modulated Signal")
plt.show()


##########demodulation###############
numtaps = 64
delays = np.arange(numtaps) # 
b1 = signal.firwin(numtaps, 0.1) # 0.1 means the cut off frequency which is normalised
rx_demod = np.empty(0)
for i in range(Nbits):
    for j in range(bit_len):
        rx_demod = np.append(rx_demod,tx_mod[i*bit_len+j]*np.cos(2*np.pi*fc*(i*bit_len+j)))# 
# the process of modulating process :s(t)*cos"*cos
plt.figure()
plt.plot(np.abs(fft.fft(rx_demod))) #
plt.title("Spectrum of Modulated Signal * cos(wct)")
plt.show()
rx_filt = signal.lfilter(b1,1,rx_demod)
rx_filt = np.append(rx_filt,-np.ones(numtaps//2))

plt.figure()
plt.plot(rx_filt)
plt.title("Signal which passed Lpf")
plt.show()

demodulated_signal = np.empty(0) # modulated signal
for i in range(Nbits):
    t = (2*i+1)*bit_len//2 + numtaps//2
    demodulated_signal = np.append(demodulated_signal,rx_filt[t] > 0.0) # threshold function

plt.figure()
plt.plot(demodulated_signal)
plt.title("The demodulated Signal")
plt.xlim(0,24)
plt.show()





