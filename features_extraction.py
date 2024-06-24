import os
import pandas as pd
import numpy as np
import scipy.signal
from config import args

def get_mean(data: pd.DataFrame) -> pd.DataFrame:
    # return mean of each column
    data = data.mean().to_frame().T
    return data

def get_std(data: pd.DataFrame) -> pd.DataFrame:
    # return standard of each column
    data = data.std().to_frame().T
    return data

def get_integral(data: pd.DataFrame) -> pd.DataFrame:
    # return integral of each column
    data = data.apply(lambda col: np.trapz(col)).to_frame().T
    return data

def get_energy(data: pd.DataFrame) -> pd.DataFrame:
    # return energy of each column
    data = np.square(data).sum().to_frame().T
    return data

def get_power(data: pd.DataFrame) -> pd.DataFrame:
    # return power of each column
    data = np.square(data).mean().to_frame().T
    return data

def get_sum(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sum().to_frame().T
    return data

def get_zero_crossing_rate(data: pd.DataFrame) -> pd.DataFrame:
    # return zero crossing rate of each column
    def get_zcr_of_one_signal(signal):
        num_crossings = np.where(np.diff(np.signbit(signal)))[0]
        zcr = len(num_crossings) / len(signal)
        return zcr
    data = data.apply(get_zcr_of_one_signal).to_frame().T
    return data

def get_peak(data: pd.DataFrame) -> pd.DataFrame:
    # return peak value of each column
    data = pd.DataFrame(data.max()).T
    return data

def get_valley(data: pd.DataFrame) -> pd.DataFrame:
    # return valley value of each column
    data = pd.DataFrame(data.min()).T
    return data

def fft(data: pd.DataFrame) -> pd.DataFrame:
    # return DFT of each column
    columns = data.columns
    data = np.fft.fft(data)
    data = pd.DataFrame(data, columns=columns)
    return data
    
def corr(data: pd.DataFrame) -> pd.DataFrame:
    # return cross correlation of each columns
    data = data.corr()
    return data

def PSD(data: pd.DataFrame, fs: int = 1, nperseg: int = 1024) -> pd.DataFrame:
    # return power spectral density of each columns
    psd = [scipy.signal.welch(data[column].values, fs=fs, nperseg=nperseg)[1] for column in data.columns]
    data = pd.DataFrame(np.array(psd).T, columns=data.columns)
    return data

def extract_time_domain_features(data: pd.DataFrame) -> pd.DataFrame:
    # return nine time domain features of each column
    peak = get_peak(data)
    vally = get_valley(data)
    average = get_mean(data)
    variance = get_std(data)
    integral = get_integral(data)
    energy = get_energy(data)
    zcr = get_zero_crossing_rate(data)
    data = pd.concat([peak, vally, average, variance, integral, energy, zcr])
    return data

def extract_frequency_domain_features(data: pd.DataFrame, fs: float) -> pd.DataFrame:
    # return two frequency domain features of each column
    def extract_frequency_domain_featrues_one_signal(signal):
        frequencies, psd = scipy.signal.welch(signal, fs=fs)
        f1, f2, f3 = 0.01, 0.25, 0.5
        P1_v = psd[(frequencies >= f1) & (frequencies <= f2)].max()
        P2_v = psd[(frequencies >= f2) & (frequencies <= f3)].max()
        return [P1_v, P2_v]
    feat = {}
    for column in data.columns:
        extracted_features = extract_frequency_domain_featrues_one_signal(data[column])
        feat[column] = extracted_features
    data = pd.DataFrame(feat)
    return data

def extract_features_nomalized(data: pd.DataFrame, fs: float = args.fs) -> pd.DataFrame:
    # freatures extraction
    time_domain_features = extract_time_domain_features(data)
    frequency_domain_features = extract_frequency_domain_features(data, fs=fs)
    data = pd.concat([time_domain_features, frequency_domain_features])
    # normalization
    row_means = data[data != 0].mean(axis=1)
    row_std = data[data != 0].std(axis=1)
    data = (data.sub(row_means, axis=0)).div(row_std, axis=0)
    return data