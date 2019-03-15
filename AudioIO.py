from __future__ import print_function
from pydub import AudioSegment
import numpy as np
import os


def readAudioFile(path):
    '''
    read .wav files
    (Fs, x) = readAudioFile(path)
    input:
        path: waf file path
    output:
        Fs: recorded frequency of wav file, for ALC dataset, this number is 44.1K
        x: array like data
    '''

    extension = os.path.splitext(path)[1]

    try:
        if extension.lower() == '.wav':
            try:
                audiofile = AudioSegment.from_wav(path)
            except:
                print("Error: file not found or other I/O error.")
                return (-1, -1)
            if audiofile.sample_width==2:
                data = np.fromstring(audiofile._data, np.int16)
            elif audiofile.sample_width==4:
                data = np.fromstring(audiofile._data, np.int32)
            else:
                return (-1, -1)
            Fs = audiofile.frame_rate
            x = []
            for chn in list(range(audiofile.channels)):
                x.append(data[chn::audiofile.channels])
            x = np.array(x).T
        else:
            print("Error in readAudioFile(): Unknown file type!")
            return (-1,-1)
    except IOError:
        print("Error: file not found or other I/O error.")
        return (-1,-1)

    if x.ndim==2:
        if x.shape[1]==1:
            x = x.flatten()

    return (Fs, x)

if __name__ == '__main__':
    path = "/Users/shuizhuyu/Desktop/ALC/AudioPhi/BLOCK10/SES1006/0061006001_h_00.wav"
    wavfile = readAudioFile(path)
    print(wavfile)
