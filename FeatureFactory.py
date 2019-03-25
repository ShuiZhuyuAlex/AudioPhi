from Wav2Img import *
from AudioIO import readAudioFile
from MixPhi import stFeatureExtraction

def Generator(path, flag={}):
    res = dict()

    if "PAA" in flag.keys():
        flag["PAA"] = False
    res["GAF"]  = GAF(path, flag["PAA"])
    res["MTF"]  = MTF(path, flag["PAA"])
    res["Scalogram"]  = Scalogram(path, flag["PAA"])

    fs, x = readAudioFile(path)
    if "win" not in flag.keys():
        flag["win"] = 0.05*fs
    if "step" not in flag.keys():
        flag["step"] = 0.025*fs
    res["MIX"] = stFeatureExtraction(path, fs, flag["win"], flag["step"])

    return res
