import pandas as pd
import librosa as lr
import numpy as np
from AudioIO import readAudioFile

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

def Scalogram(path, hop_length=128, fmin=30, bins_per_octave=32, n_bins=250):
    fs, x = readAudioFile(path)

    clip_cqt = lr.cqt(x,
                      hop_length=hop_length,
                      sr=sr,
                      fmin=fmin,
                      bins_per_octave=bins_per_octave,
                      n_bins=n_bins,
                      filter_scale=1.)

    clip_cqt_abs = np.abs(clip_cqt)
    clip_cqt_ang = np.angle(clip_cqt)

    return clip_cqt_abs,clip_cqt_ang
