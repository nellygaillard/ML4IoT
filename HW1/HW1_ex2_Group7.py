import numpy as np
import os
from subprocess import Popen
import tensorflow as tf
import time
from scipy import signal
from scipy.io import wavfile


def compute_spectrogram_bins(rate, length=0.016):
    """Compute the number of spectrogram bins.
    
    Parameters
    ----------
    - rate [float]
    - length [float]
    
    Output
    ------
    - int"""

    frame_length = rate*length
    return int(2**np.ceil(np.log(frame_length) / np.log(2.0))) // 2 + 1

def scale_tensor(t):
    """Map the -32768 to 32767 signed 16-bit values of the tensor to -1.0 to 1.0 in float.
    
    Parameters
    ----------
    - t [tf.Tensor]
    
    Output
    ------
    - tf.Tensor: scaled float version of the input tensor"""

    t = tf.cast(t,tf.float32)
    rescaled = -1 + 2 * (t + 32768) / float(32767 + 32768) 
    return tf.cast(rescaled,tf.float32)

def compute_spectrogram(audio, length=0.016, step=0.008, sampling_rate=16000):
    """Compute the spectogram of a given audio signal.
    
    Parameters:
    ----------
    - audio [tf.Tensor, dtype = int16]: audio of which the spectrogram has to be computed
    - length [float]: default is 16ms
    - frame_step [float]: default is 8ms
    
    Output
    ------
    - tf.Tensor"""

    frame_length = int(length*sampling_rate)
    frame_step = int(step*sampling_rate)

    tf_audio = scale_tensor(audio)
    
    # apply the STFT to get a waveform spectogram
    # it gets as input a float32/float64 Tensor of real-valued signals.
    stft = tf.signal.stft(tf_audio, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(stft)

    return spectrogram



def compute_mfccs(spectrogram, weight_matrix, num_coeff=10):
    """Extract the MFCC from audio signnals.
    
    Parameters
    ----------
    - spectrogram [tf.Tensor]: spectrogram of the signal to process""
    - weight_matrix [tf.Tensor]: matrix to warp linear scale spectrograms to the mel scale
    - num_coeff [int]: number of coefficients of the log-scaled Mel spectrogram
    
    Output
    ------
    - tf.Tensor: first num_coeff extracted from the log-scaled Mel spectrogram"""

    mel_spectrogram = tf.tensordot(spectrogram, weight_matrix, 1) 
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coeff]

    return mfccs

def SNR(mfcc_slow, mfcc_fast):
    """Compute the signal to noise ration between the MFCCs coefficients obtained with the fast pre-processing pipeline and the ones obtained with the slow pre-processing pipeline.
    
    Parameters
    ----------
    - mfcc_slow [tf.Tensor]: num_coeff coefficients from the slow pipeline
    - mfcc_fast [tf.Tensor]: num_coeff coefficients from the fast pipeline
    
    Output
    ------
    - float: SNR"""
    
    argument = np.linalg.norm(mfcc_slow)/(np.linalg.norm(mfcc_slow - mfcc_fast + 1.e-6))
    return 20*np.log10(argument)

if __name__ == '__main__':

    Popen('sudo sh -c "echo performance >'
    '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
    shell=True).wait()

     # OSS: the up_freq must not be larger than downrate/2 (Nyquist theorem)
    param = [15, 300, 1000, 4] # num_mel_bins, low_frequency, upper_frequency, downsampling_factor
        
    counter = 0
    acc_spec = 0
    acc_prep = 0
    acc_mfcc = 0
    acc_slow = 0
    acc_fast = 0
    acc_snr = 0

    down_rate = int(16000/param[3])

    spectrogram_bins = compute_spectrogram_bins(down_rate)
    spectrogram_bins_slow = compute_spectrogram_bins(16000)

    weight_matrix_slow = tf.signal.linear_to_mel_weight_matrix(num_mel_bins = 40, num_spectrogram_bins = spectrogram_bins_slow, sample_rate = 16000, lower_edge_hertz = 20, upper_edge_hertz = 4000) 
    weight_matrix_fast = tf.signal.linear_to_mel_weight_matrix(num_mel_bins = param[0], num_spectrogram_bins = spectrogram_bins, sample_rate = down_rate, lower_edge_hertz = param[1], upper_edge_hertz = param[2])                                                                    
    
    with os.scandir("./yes_no") as it:
        for entry in it: 
            
            # Preprocessing of the audio signal
            start_reading = time.time()
            rate, audio_original =  wavfile.read(entry.path)     
            reading_time = time.time() - start_reading

            start_slow = time.time()
            audio = audio_original.astype(np.int16) # cast to int16
            audio = tf.convert_to_tensor(audio) # create a tensor

            # generate STFT for slow pipeline
            spectrogram = compute_spectrogram(audio, sampling_rate=rate)              
            
            # SLOW MFCCs: generate MFCC with #mel_bins = 40, low_freq = 20Hz and up_freq=4000Hz, #mfccs = 10
            mfcc_slow = compute_mfccs(spectrogram, weight_matrix_slow)
            end_slow = time.time() - start_slow
            
            start_preprocessing = time.time()
            audio = signal.resample_poly(audio_original,1,param[3]) 
            audio = audio.astype(np.int16)                  
            audio = tf.convert_to_tensor(audio)                     
            end_preprocessing = time.time() - start_preprocessing

            # generate STFT for fast pipeline
            start_spec = time.time()
            spectrogram = compute_spectrogram(audio, sampling_rate=down_rate)              
            end_spec = time.time() - start_spec

            # FAST MFCCs
            start_fast_mfcc = time.time()
            mfcc_fast = compute_mfccs(spectrogram, weight_matrix_fast)
            end_fast_mfcc = time.time() - start_fast_mfcc
            
            # compute signal to noise ratio
            snr = SNR(mfcc_slow, mfcc_fast)
            
            acc_spec += end_spec
            acc_prep += end_preprocessing
            acc_mfcc += end_fast_mfcc
            acc_slow += end_slow + reading_time
            acc_fast += reading_time + end_spec + end_preprocessing + end_fast_mfcc
            acc_snr += snr
            counter += 1


    avg_slow = int(acc_slow/counter*1000)
    avg_fast = int(acc_fast/counter*1000)
    avg_snr = acc_snr/counter

    avg_spec = int(acc_spec/counter*1000)
    avg_prep = int(acc_prep/counter*1000)
    avg_mfcc = int(acc_mfcc/counter*1000)

    print(f"MFCC slow = {avg_slow} ms")
    print(f"MFCC fast = {avg_fast} ms")
    print(f"SNR = {avg_snr:.2f} dB")