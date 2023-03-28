from PIL import Image
import numpy as np
import scipy.io.wavfile as wav
import pyofdm.codec
import pyofdm.nyquistmodem
import matplotlib.pyplot as plt

# Number of total frequency samples
totalFreqSamples = 2048

# Number of useful data carriers / frequency samples
sym_slots = 1512

# QAM Order
QAMorder = 2

# Total number of bytes per OFDM symbol
nbytes = sym_slots * QAMorder // 8

# Distance of the evenly spaced pilots
distanceOfPilots = 12
pilotlist = pyofdm.codec.setpilotindex(nbytes, QAMorder, distanceOfPilots)

ofdm = pyofdm.codec.OFDM(pilotAmplitude=16/9,
                         nData=nbytes,
                         pilotIndices=pilotlist,
                         mQAM=QAMorder,
                         nFreqSamples=totalFreqSamples)


def receive(wave_file):
    samp_rate, base_signal = wav.read(wave_file)
    # append some extra zeros to the base_signal
    extra_pad_length = 60
    base_signal = np.pad(base_signal, (0, extra_pad_length), "constant")
    complex_signal = pyofdm.nyquistmodem.demod(base_signal)

    Nsig_sym = 159
    ofdm.initDecode(complex_signal, 25)
    rx_byte = np.uint8([ofdm.decode()[0] for i in range(Nsig_sym)]).ravel()
    rx_byte = 255 - rx_byte

    rx_byte = rx_byte[:60000].reshape(200, 300)
    receive_img = Image.fromarray(rx_byte)
    plt.imshow(receive_img, plt.cm.gray)
    plt.show()

    # calculate bit error ratio
    origin_img = Image.open("DC4_300x200.pgm")
    origin_img = np.array(origin_img)
    ber = np.sum(origin_img != receive_img) / origin_img.size
    print("Bit error ratio = ", ber)


# receive("ofdm44100_reverb.wav")
# receive("ofdm44100_reverb_B20.wav")
# receive("ofdm44100_reverb_B80.wav")
# receive("ofdm44100_reverb_M20.wav")
# receive("ofdm44100_reverb_M80.wav")
# receive("ofdm44100_white_noise0.8.wav")
receive("ofdm44100_white_noise0.05.wav")
receive("ofdm44100_white_noise0.015.wav")
