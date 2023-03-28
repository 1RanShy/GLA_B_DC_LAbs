'''
Date         : 2023/02/07
LastEditors  : RanShuai
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
from scipy import signal
from numpy import random

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(bool)
# tranfer student number to 24 bit ninary data
id_num = 2633609
Nbits = 24
tx_bin = bin_array(id_num, Nbits) # the original data

# The Original signal
plt.figure()
plt.title('The Original signal')
plt.plot(tx_bin)
plt.show()

s = tx_bin # copy the original signal for Differential coding
# carrier frequency
f_ideal = 1/32 # the normalised ideal frequency 
# bit length
bit_len = 64

# turn on or off the  differential coding operation
flagDiffCoding = True
if (flagDiffCoding):
    #Differential Coding of tx_bin
    tx_diff = np.zeros(1, dtype='bool')
    for i in range(Nbits):
        tx_diff = np.append(tx_diff, tx_diff[i]^s[i])
    Nbits = Nbits+1
else:
    tx_diff = tx_bin
    

# low-pass filter
numtaps = 64 # taps
b1 = np.flip(signal.firwin(numtaps, 0.1))    # the filter coefficients


clock = np.array([1.0,0.0]) # the original value of  VCO Cosoutput and Sine_output
# Add some randome noise to the ideal carrier frequency
f_noise =  f_ideal*(1.+0.02*(random.rand()-0.5))
# Add some phase error to the ideal carrier frequency to simultae the error carrier frequency
ph_c = 2*np.pi*random.rand()
volt = 0.0 # The error to drive the VCO
vout = np.array(volt)# The error to drive the VCO
# out_array of clock(recovery clock)

cout = clock[0] # the cos part which is not passed LPF
# out_array of reference clock
rout = np.cos(ph_c)
# Data outpur of c_output
dout = np.empty(0)

mixed = np.zeros((2,numtaps))
for i in range(0, bit_len*Nbits + numtaps//2):
    #modulation equation            
    mixed[0,:] = np.append(mixed[0,1:],clock[0]*(2*tx_diff[(i//bit_len)%Nbits]-1)*np.cos(ph_c+2*np.pi*f_noise*i))
    mixed[1,:] = np.append(mixed[1,1:],-clock[1]*(2*tx_diff[(i//bit_len)%Nbits]-1)*np.cos(ph_c+2*np.pi*f_noise*i))            
    # Filtering of mixed data
    lpmixed = [np.sum(b1*mixed[j,:]) for j in range(2)]
    volt = lpmixed[0]*lpmixed[1]
            
    c = np.cos(2*np.pi*f_ideal*(1.+0.25*volt))
    s = np.sin(2*np.pi*f_ideal*(1.+0.25*volt))
    clock = np.matmul(np.array([[c, -s], [s, c]]), clock)
            
    vout = np.append(vout, volt)
    cout = np.append(cout, clock[0])
    rout = np.append(rout, np.cos(ph_c+2*np.pi*f_noise*i)) # the noise carrier frequncy red
    dout = np.append(dout, lpmixed[0]) # the adapting frequency which should be exactly the same as the noise carrier frequency  blue

plt.figure()
plt.title('Voltage/Error driving the VCO')
plt.axhline(f_noise,color='r')
plt.plot(vout, color='b')
plt.show()

plt.figure()
plt.title('Clock_Output Blue|Reference_Output Red')
plt.plot(cout,color='b')
plt.plot(rout, color='r')
plt.show()

plt.figure()
plt.title('Signal before thresholding')
plt.plot(dout,color='b')
plt.show()

# two parts which do or not do the differential coding 
# This will cause the reverse demodulated signal
if (flagDiffCoding):
    rx_diff = np.empty(0)
    for i in range(Nbits):
        #select an appropriate sample point
        k = (2*i+1)*bit_len//2 +numtaps//2
        rx_diff= np.append(rx_diff, np.heaviside(dout[k],0))

    rx_bin = np.empty(0, dtype='bool')
    Nbits = Nbits-1
    for i in range(Nbits):
        rx_bin = np.append(rx_bin, rx_diff[i].astype(bool)^rx_diff[i+1].astype(bool))

    # print(rx_bin)
    plt.figure()
    plt.title('Demodulated signal with differential encoding')
    plt.plot(rx_bin) 
    plt.show()
else:
    rx_bin = np.empty(0, dtype='bool')
    for i in range(0,Nbits):
        t = (2*i+1)*bit_len//2 +numtaps//2
        rx_bin = np.append(rx_bin, np.heaviside(dout[t],0))

# judge if the demodulated signal is the same as the original signal
# if not draw the orange  wrong pic
    if ((rx_bin != tx_bin).any()):
        plt.figure()
        plt.title('Demodulated signal without differential coding')
        plt.plot(rx_bin, color='purple')
        plt.show()
    else:
        plt.figure()
        plt.title('Demodulated signal without differential coding')
        plt.plot(rx_bin)
        plt.show()