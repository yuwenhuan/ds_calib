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

def myfft(v,fs,signal_bin_num,OSR,plot,ylow,des):
    print v
    N = v.size
    w = ds_hann(N)

    nb = 3;
    w1 = np.linalg.norm(w,1)
    w2 = np.linalg.norm(w,2)
    NBW = (w2/w1)^2
    V = np.fft(w*v)/(w1/2)
    
    # integrated noise
    for ii in range(length(V)):
        if ii < 2: # dc signal
            intV[ii] = 0
        elif ((ii -1 > signal_bin_num + 1) or (ii -1 < signal_bin_num - 1)): # not signal bin
            intV[ii] = np.sqrt(np.abs(intV[ii-1])^2 + np.abs(V[ii])^2)
        else:
            intV[ii] = intV[ii-1]

    fstep = fs/N
    f = np.arange(0,(N/2-1)*fstep,fstep)
    if plot == 1:
        # plot time domain output
        #figure;stairs(v);grid on;
        
        # plot frequency domain output
        plt.figure(2)
        plt.semilogx(f,20*np.log10(np.abs(V[1:N/2])),'LineWidth',2)
        plt.hold(True);
        plt.semilogx(f,20*log10(abs(intV[1:N/2])),'r--','LineWidth',2)
        # plot OSR limit
        plt.semilogx( [fs/2/OSR,fs/2/OSR],[0,ylow],'k','LineWidth',2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dBFS)')
        plt.axis([0,fs/2,ylow,0])
        plt.title(des)
        plt.legend('Output spectrum','Integrated noise')
        plt.grid(True)

    #[snr,sndr,hd2,hd3] =mysnr(V,signal_bin_num,OSR)
    return V