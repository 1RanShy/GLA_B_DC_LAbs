#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:30:04 2023

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
id_num = 2635088
Nbits = 24
tx_bin = bin_array(id_num, Nbits) # 转换后的24位二进制学号
bit_len = 16 #
fc = 0.125 # normalized frequency
s = np.copy(tx_bin) # s就是原始信号
tx_mod = np.empty(0) # 被调制之后的信号

plt.figure()
plt.plot(tx_bin)
plt.title("Original signal")
plt.show()
############### modulation###############
for i in range(Nbits):
    for j in range(bit_len):
        tx_mod = np.append(tx_mod,s[i]*np.cos(2*np.pi*fc*(i*bit_len+j)))# 把信号追加到tx_mod中

plt.figure()
plt.plot(tx_mod)
plt.title("The modulated Signal")
plt.show()

plt.figure()
plt.plot(np.abs(fft.fft(tx_mod))) #因为是傅立叶变换是 complex quantity 复数 方便在spectrum analyser上查看
plt.title("The Spectrum of modulated Signal")
plt.show()


##########demodulation###############
numtaps = 64
mydelays = np.arange(numtaps) # 为了画载波的波形
b1 = signal.firwin(numtaps, 0.1) # 滤波器的系数
#mixed = np.zeros(numtaps)

###画出载波频率###
fig, axs = plt.subplots()
axs.bar(mydelays,b1)
axs.axhline()
axs.set_title("Low pass")
plt.show()

###画传递函数的 幅频响应###
w1, h1 = signal.freqz(b1)
plt.plot(w1/2/np.pi, 20*np.log10(np.abs(h1))) #频率
plt.ylabel("Amplitude Response/dB")
plt.xlabel("Frequency/sample rate")
plt.grid()
plt.title("Digital filter frequency response")
plt.show()

### 画半解调后的信号###
rx_mixed = tx_mod*np.heaviside(tx_mod,0)#这是为了整流 转为二进制
rx_lpf = signal.lfilter(b1, 1, rx_mixed)# 这是解调后的信号
plt.figure()
plt.plot(rx_lpf)
plt.title("The signal which passed Lpf")
plt.show()

### 绘制解调后的信号 ###
rx_bin = np.empty(0)
for i in range(0,Nbits):
    t = (2*i+1)*bit_len//2
    rx_bin = np.append(rx_bin,rx_lpf[t] > 0.1)
plt.figure()
plt.plot(rx_bin)
plt.title("The demodulated Signal")
plt.show()
