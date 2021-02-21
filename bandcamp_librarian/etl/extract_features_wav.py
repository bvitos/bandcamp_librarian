#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts 92 pyAudio and Essentia features from a .wav (22050 Hz mono) file
Based on pyGenreClf / A. Caparrini
https://github.com/Caparrini/pyGenreClf
"""


from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction
from pyAudioAnalysis.MidTermFeatures import beat_extraction
import pandas as pd
import numpy as np

import essentia.standard as es
import time


def file_extract_essentia(audio_file):
    """
    Essentia feature extraction / tailored code from pyGenreClf
    
    Returns: features
    """
    rhythm_feats = ["rhythm.bpm",
                    "rhythm.bpm_histogram_first_peak_bpm",
                    "rhythm.bpm_histogram_first_peak_weight",
                    "rhythm.bpm_histogram_second_peak_bpm",
                    "rhythm.bpm_histogram_second_peak_spread",
                    "rhythm.bpm_histogram_second_peak_weight",
                    "rhythm.danceability",
                    "rhythm.beats_loudness.mean",
                    "rhythm.beats_loudness.stdev",
                    "rhythm.onset_rate",
                    "rhythm.beats_loudness_band_ratio.mean",
                    "rhythm.beats_loudness_band_ratio.stdev"
                    ]

    features_total, features_frames = es.MusicExtractor()(audio_file)

    features = []
    for i in range(0,len(rhythm_feats)-2):
        features.append(features_total[rhythm_feats[i]])

    for i in range(len(rhythm_feats)-2,len(rhythm_feats)):
        bands = features_total[rhythm_feats[i]]
        for j in range(0,len(bands)):
            features.append(bands[j])

    return np.array(features)



def feature_extract_pyAudio(file_path, mid_window, mid_step, short_window, short_step):
    """
    pyAudio feature extraction / tailored code from pyGenreClf
    
    Returns: features
    """
    mid_term_features = np.array([])
    process_times = []
    mid_feature_names = []   
    sampling_rate, signal = audioBasicIO.read_audio_file(file_path)
    t1 = time.time()        
    signal = audioBasicIO.stereo_to_mono(signal)
    if signal.shape[0] < float(sampling_rate)/5:
        print("  (AUDIO FILE TOO SMALL - SKIPPING)")
    else:
        mid_features, short_features, mid_feature_names = \
            mid_feature_extraction(signal, sampling_rate,
                                   round(mid_window * sampling_rate),
                                   round(mid_step * sampling_rate),
                                   round(sampling_rate * short_window),
                                   round(sampling_rate * short_step))
        beat, beat_conf = beat_extraction(short_features, short_step)
        
        mid_features = np.transpose(mid_features)
        mid_features = mid_features.mean(axis=0)
        # long term averaging of mid-term statistics
        if (not np.isnan(mid_features).any()) and \
                (not np.isinf(mid_features).any()):
            mid_features = np.append(mid_features, beat)
            mid_features = np.append(mid_features, beat_conf)
            if len(mid_term_features) == 0:
                # append feature vector
                mid_term_features = mid_features
            else:
                mid_term_features = np.vstack((mid_term_features, mid_features))
            t2 = time.time()
            duration = float(len(signal)) / sampling_rate
            process_times.append((t2 - t1) / duration)
        if len(process_times) > 0:
            print("Feature extraction complexity ratio: "
                  "{0:.1f} x realtime".format((1.0 / np.mean(np.array(process_times)))))
    return mid_term_features, mid_feature_names



def extract_features_wav_file(file_path):    
    """
    Extracts the 92 audio features
    
    Returns: 1-row dataframe with features
    """
    data = []
    feature_values, feature_names = feature_extract_pyAudio(file_path, 1, 1, 0.05, 0.05)
    feature_names = feature_names + ['BPM','BPMConf']
    feature_names = feature_names + ['71-BPMessentia','72-bpm_histogram_first_peak_bpm','73-bpm_histogram_first_peak_weight','74-bpm_histogram_second_peak_bpm','75-bpm_histogram_second_peak_spread','76-bpm_histogram_second_peak_weight','77-danceability','78-beats_loudness.mean','79-beats_loudness.stdev','80-onset_rate','81-beats_loudness_band_ratio.mean1','82-beats_loudness_band_ratio.mean2','83-beats_loudness_band_ratio.mean3','84-beats_loudness_band_ratio.mean4','85-beats_loudness_band_ratio.mean5','86-beats_loudness_band_ratio.mean6','87-beats_loudness_band_ratio.stdev1','88-beats_loudness_band_ratio.stdev2','89-beats_loudness_band_ratio.stdev3','90-beats_loudness_band_ratio.stdev4','91-beats_loudness_band_ratio.stdev5','92-beats_loudness_band_ratio.stdev6']
    rhythm_feats = file_extract_essentia(file_path)
    data.append(feature_values.tolist() + rhythm_feats.tolist())
    df = pd.DataFrame(np.array(data), columns=feature_names)
    df.drop([col for col in df.columns if 'delta' in col],axis=1,inplace=True)          # drop delta values
    df.columns = ['1-ZCRm', '2-Energym', '3-EnergyEntropym', '4-SpectralCentroidm', '5-SpectralSpreadm', '6-SpectralEntropym', '7-SpectralFluxm', '8-SpectralRolloffm', '9-MFCCs1m', '10-MFCCs2m', '11-MFCCs3m', '12-MFCCs4m', '13-MFCCs5m', '14-MFCCs6m', '15-MFCCs7m', '16-MFCCs8m', '17-MFCCs9m', '18-MFCCs10m', '19-MFCCs11m', '20-MFCCs12m', '21-MFCCs13m', '22-ChromaVector1m', '23-ChromaVector2m', '24-ChromaVector3m', '25-ChromaVector4m', '26-ChromaVector5m', '27-ChromaVector6m', '28-ChromaVector7m', '29-ChromaVector8m', '30-ChromaVector9m', '31-ChromaVector10m', '32-ChromaVector11m', '33-ChromaVector12m', '34-ChromaDeviationm', '35-ZCRstd', '36-Energystd', '37-EnergyEntropystd', '38-SpectralCentroidstd', '39-SpectralSpreadstd', '40-SpectralEntropystd', '41-SpectralFluxstd', '42-SpectralRolloffstd', '43-MFCCs1std', '44-MFCCs2std', '45-MFCCs3std', '46-MFCCs4std', '47-MFCCs5std', '48-MFCCs6std', '49-MFCCs7std', '50-MFCCs8std', '51-MFCCs9std', '52-MFCCs10std', '53-MFCCs11std', '54-MFCCs12std', '55-MFCCs13std', '56-ChromaVector1std', '57-ChromaVector2std', '58-ChromaVector3std', '59-ChromaVector4std', '60-ChromaVector5std', '61-ChromaVector6std', '62-ChromaVector7std', '63-ChromaVector8std', '64-ChromaVector9std', '65-ChromaVector10std', '66-ChromaVector11std', '67-ChromaVector12std', '68-ChromaDeviationstd', '69-BPM', '70-BPMconf', '71-BPMessentia','72-bpm_histogram_first_peak_bpm','73-bpm_histogram_first_peak_weight','74-bpm_histogram_second_peak_bpm','75-bpm_histogram_second_peak_spread','76-bpm_histogram_second_peak_weight','77-danceability','78-beats_loudness.mean','79-beats_loudness.stdev','80-onset_rate','81-beats_loudness_band_ratio.mean1','82-beats_loudness_band_ratio.mean2','83-beats_loudness_band_ratio.mean3','84-beats_loudness_band_ratio.mean4','85-beats_loudness_band_ratio.mean5','86-beats_loudness_band_ratio.mean6','87-beats_loudness_band_ratio.stdev1','88-beats_loudness_band_ratio.stdev2','89-beats_loudness_band_ratio.stdev3','90-beats_loudness_band_ratio.stdev4','91-beats_loudness_band_ratio.stdev5','92-beats_loudness_band_ratio.stdev6']
    # the final column names replicate the column names of Caparrini's Beatport datasets; see https://www.kaggle.com/caparrini/electronic-music-features-201802-beatporttop100
    return df

