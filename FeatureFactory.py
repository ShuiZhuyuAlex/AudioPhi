import os
import argparse
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct

'''

'''


class Wav2Img(object):

    def __init__(self, flag, index):
        self.flag = flag
        self.index = index

    def PAASmoothing(data, now=64, opw):
        if now == None:
            now = int(len(data) / opw)
        if opw == None:
            opw = int(len(data) / now)
        temp = [sum(data[i * opw : (i + 1) * opw]) / float(opw) for i in range(now)]
        return np.array(temp)

    def Rescale(data):
        maxval = max(data)
        minval = min(data)
        gap = float(maxval-minval)
        return [(each-minval)/gap*2-1 for each in data]

    def GAF(path, PAA=False):
        fs, x = readAudioFile(path)
        std_data = Rescale(x)

        datacos = np.array(std_data)
        datasin = np.sqrt(1-np.array(std_data)**2)

        paalistcos = PAASmoothing(std_data, s, None)
        paalistsin = np.sqrt(1-paalistcos**2)

        datacos = np.matrix(datacos)
        datasin = np.matrix(datasin)

        paalistcos = np.matrix(paalistcos)
        paalistsin = np.matrix(paalistsin)

        gasf_paamatrix = np.array(paalistcos.T*paalistcos-paalistsin.T*paalistsin)
        gasf_matrix = np.array(datacos.T*datacos-datasin.T*datasin)

        gadf_paamatrix = np.array(paalistsin.T*paalistcos-paalistcos.T*paalistsin)
        gadf_matrix = np.array(datasin.T*datacos - datacos.T*datasin)

        if PAA:
            return gasf_paamatrix,gadf_paamatrix
        else:
            return gasf_matrix,gadf_matrix

    def MTF(path, Q=16, PAA=False):
        fs, x = readAudioFile(path)
        std_data = Rescale(x)

        paalist = PAASmoothing(std_data, s, None)

        mat, matindex, level = QMeq(std_data, Q)
        paamatindex = paaMarkovMatrix(paalist, level)

        column = []
        paacolumn = []
        for p in range(len(std_data)):
            for q in range(len(std_data)):
                column.append(mat[matindex[p]][matindex[(q)]])

        for p in range(s):
            for q in range(s):
                paacolumn.append(mat[paamatindex[p]][paamatindex[(q)]])

        columnmatrix = np.array(column).reshape(len(std_data),len(std_data))
        paacolumn = np.array(paacolumn)
        if PAA:
            return paacolumn
        else:
            return columnmatrix


class FeatureFactory(object):
    '''
    '''
    def __init__(self, feature_list, index):
        self.feature_list = feature_list
        self.index = index
        self.eps = 0.00000001

    def _stZCR(self, frame):
        """Computes zero crossing rate of frame"""
        count = len(frame)
        countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        return (np.float64(countZ) / np.float64(count-1.0))


    def _stEnergy(self, frame):
        """Computes signal energy of frame"""
        return np.sum(frame ** 2) / np.float64(len(frame))


    def _stEnergyEntropy(self, frame, n_short_blocks=10):
        """Computes entropy of energy"""
        Eol = np.sum(frame ** 2)
        L = len(frame)
        sub_win_len = int(np.floor(L / n_short_blocks))
        if L != sub_win_len * n_short_blocks:
                frame = frame[0:sub_win_len * n_short_blocks]
        sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

        # Compute normalized sub-frame energies:
        s = np.sum(sub_wins ** 2, axis=0) / (Eol + self.eps)

        # Compute entropy of the normalized sub-frame energies:
        Entropy = -np.sum(s * np.log2(s + self.eps))
        return Entropy


    """ Frequency-domain audio features """


    def _stSpectralCentroidAndSpread(self, X, fs):
        """Computes spectral centroid of frame (given abs(FFT))"""
        ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

        Xt = X.copy()
        Xt = Xt / Xt.max()
        NUM = np.sum(ind * Xt)
        DEN = np.sum(Xt) + self.eps

        # Centroid:
        C = (NUM / DEN)

        # Spread:
        S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

        # Normalize:
        C = C / (fs / 2.0)
        S = S / (fs / 2.0)

        return (C, S)


    def _stSpectralEntropy(self, X, n_short_blocks=10):
        """Computes the spectral entropy"""
        L = len(X)                         # number of frame samples
        Eol = np.sum(X ** 2)            # total spectral energy

        sub_win_len = int(np.floor(L / n_short_blocks))   # length of sub-frame
        if L != sub_win_len * n_short_blocks:
            X = X[0:sub_win_len * n_short_blocks]

        sub_wins = X.reshape(sub_win_len, n_short_blocks, order='F').copy()  # defi_ne sub-frames (using matrix reshape)
        s = np.sum(sub_wins ** 2, axis=0) / (Eol + self.eps)                      # compute spectral sub-energies
        En = -np.sum(s*np.log2(s + self.eps))                                    # compute spectral entropy

        return En


    def _stSpectralFlux(self, X, X_prev):
        """
        Computes the spectral flux feature of the current frame
        ARGUMENTS:
            X:            the abs(fft) of the current frame
            X_prev:        the abs(fft) of the previous frame
        """
        # compute the spectral flux as the sum of square distances:
        sumX = np.sum(X + self.eps)
        sumPrevX = np.sum(X_prev + self.eps)
        F = np.sum((X / sumX - X_prev/sumPrevX) ** 2)

        return F


    def _stSpectralRollOff(self, X, c, fs):
        """Computes spectral roll-off"""
        totalEnergy = np.sum(X ** 2)
        fftLength = len(X)
        Thres = c*totalEnergy
        # Ffind the spectral rolloff as the frequency position
        # where the respective spectral energy is equal to c*totalEnergy
        CumSum = np.cumsum(X ** 2) + self.eps
        [a, ] = np.nonzero(CumSum > Thres)
        if len(a) > 0:
            mC = np.float64(a[0]) / (float(fftLength))
        else:
            mC = 0.0
        return (mC)


    def _stHarmonic(self, frame, fs):
        """
        Computes harmonic ratio and pitch
        """
        M = np.round(0.016 * fs) - 1
        R = np.correlate(frame, frame, mode='full')

        g = R[len(frame)-1]
        R = R[len(frame):-1]

        # estimate m0 (as the first zero crossing of R)
        [a, ] = np.nonzero(np.diff(np.sign(R)))

        if len(a) == 0:
            m0 = len(R)-1
        else:
            m0 = a[0]
        if M > len(R):
            M = len(R) - 1

        Gamma = np.zeros((M), dtype=np.float64)
        CSum = np.cumsum(frame ** 2)
        Gamma[m0:M] = R[m0:M] / (np.sqrt((g * CSum[M:m0:-1])) + self.eps)

        ZCR = stZCR(Gamma)

        if ZCR > 0.15:
            HR = 0.0
            f0 = 0.0
        else:
            if len(Gamma) == 0:
                HR = 1.0
                blag = 0.0
                Gamma = np.zeros((M), dtype=np.float64)
            else:
                HR = np.max(Gamma)
                blag = np.argmax(Gamma)

            # Get fundamental frequency:
            f0 = fs / (blag + self.eps)
            if f0 > 5000:
                f0 = 0.0
            if HR < 0.1:
                f0 = 0.0

        return (HR, f0)


    def _mfccInitFilterBanks(self, fs, nfft):
        """
        Computes the triangular filterbank for MFCC computation
        (used in the stFeatureExtraction function before the stMFCC function call)
        This function is taken from the scikits.talkbox library (MIT Licence):
        https://pypi.python.org/pypi/scikits.talkbox
        """

        # filter bank params:
        lowfreq = 133.33
        linsc = 200/3.
        logsc = 1.0711703
        numLinFiltTotal = 13
        numLogFilt = 27

        if fs < 8000:
            nlogfil = 5

        # Total number of filters
        nFiltTotal = numLinFiltTotal + numLogFilt

        # Compute frequency points of the triangle:
        freqs = np.zeros(nFiltTotal+2)
        freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
        freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** np.arange(1, numLogFilt + 3)
        heights = 2./(freqs[2:] - freqs[0:-2])

        # Compute filterbank coeff (in fft domain, in bins)
        fbank = np.zeros((nFiltTotal, nfft))
        nfreqs = np.arange(nfft) / (1. * nfft) * fs

        for i in range(nFiltTotal):
            lowTrFreq = freqs[i]
            cenTrFreq = freqs[i+1]
            highTrFreq = freqs[i+2]

            lid = np.arange(np.floor(lowTrFreq * nfft / fs) + 1,
                               np.floor(cenTrFreq * nfft / fs) + 1,
                                           dtype=np.int)
            lslope = heights[i] / (cenTrFreq - lowTrFreq)
            rid = np.arange(np.floor(cenTrFreq * nfft / fs) + 1,
                                           np.floor(highTrFreq * nfft / fs) + 1,
                                           dtype=np.int)
            rslope = heights[i] / (highTrFreq - cenTrFreq)
            fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
            fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

        return fbank, freqs


    def _stMFCC(self, X, fbank, n_mfcc_feats):
        """
        Computes the MFCCs of a frame, given the fft mag

        ARGUMENTS:
            X:        fft magnitude abs(FFT)
            fbank:    filter bank (see mfccInitFilterBanks)
        RETURN
            ceps:     MFCCs (13 element vector)

        Note:    MFCC calculation is, in general, taken from the
                 scikits.talkbox library (MIT Licence),
        #    with a small number of modifications to make it more
             compact and suitable for the pyAudioAnalysis Lib
        """

        mspec = np.log10(np.dot(X, fbank.T)+self.eps)
        ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:n_mfcc_feats]
        return ceps


    def _stChromaFeaturesInit(self, nfft, fs):
        """
        This function initializes the chroma matrices used in the calculation of the chroma features
        """
        freqs = np.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])
        Cp = 27.50
        nChroma = np.round(12.0 * np.log2(freqs / Cp)).astype(int)

        nFreqsPerChroma = np.zeros((nChroma.shape[0], ))

        uChroma = np.unique(nChroma)
        for u in uChroma:
            idx = np.nonzero(nChroma == u)
            nFreqsPerChroma[idx] = idx[0].shape

        return nChroma, nFreqsPerChroma


    def _stChromaFeatures(self, X, fs, nChroma, nFreqsPerChroma):
        #TODO: 1 complexity
        #TODO: 2 bug with large windows

        chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D',
                       'D#', 'E', 'F', 'F#', 'G', 'G#']
        spec = X**2
        if nChroma.max()<nChroma.shape[0]:
            C = np.zeros((nChroma.shape[0],))
            C[nChroma] = spec
            C /= nFreqsPerChroma[nChroma]
        else:
            I = np.nonzero(nChroma>nChroma.shape[0])[0][0]
            C = np.zeros((nChroma.shape[0],))
            C[nChroma[0:I-1]] = spec
            C /= nFreqsPerChroma
        finalC = np.zeros((12, 1))
        newD = int(np.ceil(C.shape[0] / 12.0) * 12)
        C2 = np.zeros((newD, ))
        C2[0:C.shape[0]] = C
        C2 = C2.reshape(int(C2.shape[0]/12), 12)
        finalC = np.matrix(np.sum(C2, axis=0)).T
        finalC /= spec.sum()
        return chromaNames, finalC


    def _stFeatureExtraction(self, signal, fs, win, step):

        """
        This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
        This results to a sequence of feature vectors, stored in a np matrix.

        ARGUMENTS
            signal:       the input signal samples
            fs:           the sampling freq (in Hz)
            win:          the short-term window size (in samples)
            step:         the short-term window step (in samples)
        RETURNS
            st_features:   a np array (n_feats(34) x numOfShortTermWindows)
            0: zcr
            1: energy
            2: energy_entropy
            3: spectral_centroid
            4: spectral_spread
            5: spectral_entropy
            6: spectral_flux
            7: spectral_rolloff
            8-20: mfcc
            21-33: chroma
        """

        win = int(win)
        step = int(step)

        # Signal normalization
        signal = np.double(signal)

        signal = signal / (2.0 ** 15)
        DC = signal.mean()
        MAX = (np.abs(signal)).max()
        signal = (signal - DC) / (MAX + 0.0000000001)

        N = len(signal)                                # total number of samples
        cur_p = 0
        count_fr = 0
        nFFT = int(win / 2)

        [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
        nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, fs)

        n_time_spectral_feats = 8
        n_harmonic_feats = 0
        n_mfcc_feats = 13
        n_chroma_feats = 13
        n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats + n_chroma_feats
        st_features = []
        while (cur_p + win - 1 < N):                        # for each short-term window until the end of signal
            count_fr += 1
            x = signal[cur_p:cur_p+win]                    # get current window
            cur_p = cur_p + step                           # update window position
            X = abs(fft(x))                                  # get fft magnitude
            X = X[0:nFFT]                                    # normalize fft
            X = X / len(X)
            if count_fr == 1:
                X_prev = X.copy()                             # keep previous fft mag (used in spectral flux)
            curFV = np.zeros((n_total_feats, 1))
            curFV[0] = stZCR(x)                              # zero crossing rate
            curFV[1] = stEnergy(x)                           # short-term energy
            curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
            [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, fs)    # spectral centroid and spread
            curFV[5] = stSpectralEntropy(X)                  # spectral entropy
            curFV[6] = stSpectralFlux(X, X_prev)              # spectral flux
            curFV[7] = stSpectralRollOff(X, 0.90, fs)        # spectral rolloff
            curFV[n_time_spectral_feats:n_time_spectral_feats+n_mfcc_feats, 0] = \
                stMFCC(X, fbank, n_mfcc_feats).copy()    # MFCCs
            chromaNames, chromaF = stChromaFeatures(X, fs, nChroma, nFreqsPerChroma)
            curFV[n_time_spectral_feats + n_mfcc_feats:
                  n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
                chromaF
            curFV[n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
                chromaF.std()
            st_features.append(curFV)
            # delta features
            '''
            if count_fr>1:
                delta = curFV - prevFV
                curFVFinal = np.concatenate((curFV, delta))
            else:
                curFVFinal = np.concatenate((curFV, curFV))
            prevFV = curFV
            st_features.append(curFVFinal)
            '''
            # end of delta
            X_prev = X.copy()

        st_features = np.concatenate(st_features, 1)
        return st_features

        @staticmethod
        def generator(self, feature_list):
            for feature in self.feature_list:
                if feature == "scalogram":
                    scalogram = Wav2Img.Scalogram()
                    pass
                elif feature == "gaf":
                    gaf = Wav2Img.GAF()
                    mtf = Wav2Img.MTF()
                    pass
                else:
                    matrix = self.stFeatureExtraction()
                    if feature == "spectral":
                        spectral = matrix[0:8]
                        #0-7
                        pass
                    elif feature == "mfcc":
                        mfcc = matrix[8:21]
                        #8-20
                        pass
                    elif feature == "chroma":
                        chroma = matrix[22:34]
                        pass
                    elif feature == "logmel_spectrogram":
                        pass
                    else:
                        #raise error
                        pass

        def Write(self):
            pass

        def Run(self):
            pass
