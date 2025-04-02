import numpy as np
from numpy import fft

def ISFFT(inSig):
    """
    Performs Inverse Symplectic Fast Fourier Transform
    
    Input arguments:
    inSig: Input N x M matrix to be transformed
    
    Function returns:
    outSig: Output N x M matrix of doppler-Delay domain symbols
    """
    N, M = inSig.shape  # Calculate N and M
    outSig = np.sqrt(N / M) * np.fft.fft(np.fft.ifft(inSig, axis=0), axis=1)  # Apply inverse transform
    return outSig


def SFFT(inSig):
    """
    Performs Symplectic Fast Fourier Transform
    
    Input arguments:
    inSig: Input N x M matrix to be transformed
    
    Function returns:
    outSig: Output N x M matrix of doppler-Delay domain symbols
    """
    N, M = inSig.shape  # Calculate N and M
    outSig = np.sqrt(M / N) * np.fft.ifft(np.fft.fft(inSig, axis=0), axis=1)  # Apply transform
    return outSig
