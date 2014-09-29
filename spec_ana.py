import numpy as np
import matplotlib.pyplot as plt

def ds_hann(n):
    """A Hann window of length :math:`n`. 

    The Hann window, aka *the raised cosine window*, is defined as:

    .. math::

        w(x) = 0.5\\ \\left(1 - cos\\left(\\frac{2 \\pi x}{n}\\right) \\right)

    This windowing function does not smear tones located exactly in a bin.

    **Parameters:**

    n : integer
        The window length, in number of samples.

    **Returns:**

    w : 1d nd_array
        The Hann window.

    .. note:: Functionally equivalent to numpy's ``hanning()``, provided
        to ease porting of code from MATLAB. Also, we take care always to
        return an array of dimensions ``(n,)`` and type ``float_``.

    .. plot::

      import pylab as plt
      from deltasigma import ds_hann
      x = ds_hann(100)
      plt.figure(figsize=(12, 5))
      plt.plot(x, 'o-')
      ax = plt.gca()
      ax.set_ylim(0, 1.02)
      plt.grid(True)
      plt.title("100-samples Hann window")
      plt.xlabel("Sample #")
      plt.ylabel("Value")

    """
    x = np.arange(n, dtype='float_')
    return .5*(1 - np.cos(2*np.pi*x/n))


def snr_calc(V,signal_bin_num,OSR):
    """
    function to calculate SNR

    Input:
    V: fft spectrum
    signal_bin_num: the fft bin number for signal (starts from 0 for DC)
    OSR: oversampling ratio
    
    Output:
    snr
    sndr
    hd2
    hd3
    """
    N = V.size
    nb = 3
    N_in_band = N/(2*OSR)
    signal_bins = np.arange(signal_bin_num - (nb-1)/2,signal_bin_num + (nb-1)/2 + 1)
    d2_bins = np.arange(2*signal_bin_num - (nb-1)/2,2*signal_bin_num + (nb-1)/2 + 1)
    d3_bins = np.arange(3*signal_bin_num - (nb-1)/2,3*signal_bin_num + (nb-1)/2 + 1)
    inband_bins = np.arange(2,N_in_band,1)
    noise_bins = np.setdiff1d(inband_bins,signal_bins)
    harmonic_bins = np.empty([0])
    for ii in range(2,N_in_band/signal_bin_num + 1):
        harmonic_bins = np.append(harmonic_bins,np.arange(ii * signal_bin_num - (nb-1)/2,ii* signal_bin_num + (nb-1)/2 + 1))
    noise_no_distortion_bins = np.setdiff1d(noise_bins,harmonic_bins)
    
    S = 0
    for ii in signal_bins:
        S = S + np.square(abs(V[ii]))

    D2 = 0
    for ii in d2_bins:
        D2 = D2 + np.square(abs(V[ii]))

    D3 = 0
    for ii in d3_bins:
        D3 = D3 + np.square(abs(V[ii]))

    N = 0
    for ii in noise_no_distortion_bins:
        N = N + np.square(abs(V[ii]))

    D = 0
    for ii in harmonic_bins:
        D = D + np.square(abs(V[ii]))
            
    snr = 10*np.log10(S/N)
    sndr = 10*np.log10(S/(N+D))
    hd2 = 10*np.log10(D2/S)
    hd3 = 10*np.log10(D3/S)
    return (snr,sndr,hd2,hd3)

def fft(v,fs,signal_bin_num,OSR,plot,ylow,des):
    N = v.size
    w = ds_hann(N)

    #nb = 3;
    w1 = np.linalg.norm(w,1)
    #w2 = np.linalg.norm(w,2)
    #NBW = np.square((w2/w1))
    V = np.fft.fft(w*v)/(w1/2)
    
    # integrated noise
    intV = np.zeros(V.size)
    for ii in range(V.size):
        if ii < 2: # dc signal
            intV[ii] = 1e-9
        elif ((ii  > signal_bin_num + 1) or (ii  < signal_bin_num - 1)): # not signal bin
            intV[ii] = np.sqrt(np.square(np.abs(intV[ii-1])) + np.square(np.abs(V[ii])))
        else:
            intV[ii] = intV[ii-1]

    fstep = fs/N
    f = np.arange(0,(N/2-1)*fstep,fstep)
    snr,sndr,hd2,hd3 = snr_calc(V,signal_bin_num,OSR)
    if plot == 1:
        # plot time domain output
        #figure;stairs(v);grid on;
        
        # plot frequency domain output
        fig2 = plt.figure(2)
        p1, = plt.semilogx(f,20*np.log10(np.abs(V[1:N/2])))
        plt.hold(True);
        p2, = plt.semilogx(f,20*np.log10(abs(intV[1:N/2])))
        # plot OSR limit
        plt.semilogx( [fs/2/OSR,fs/2/OSR],[0,ylow])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dBFS)')
        plt.axis([0,fs/2,ylow,0])
        plt.title(des)
        plt.legend([p1,p2],['Output spectrum','Integrated noise'])
        plt.grid(True)
        plt.show()
        txt = 'SNR = %4.1f dB\nSNDR= %4.1f dB\nHD2= %4.1f dB\nHD3= %4.1f dB' % (snr,sndr,hd2,hd3)
        plt.text(2e4, -155, txt, fontsize=15,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
    return V

