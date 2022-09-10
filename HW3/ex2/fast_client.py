import tensorflow as tf
import tensorflow.lite as tflite
import os
from tqdm import tqdm
import time
from scipy import signal
import pandas as pd
import base64
import datetime
import json
import requests
import sys
import numpy as np
import itertools



# definition of the fast preprocessing function

def preprocess_with_mfcc_fast(audio_binary, length, step, weight_matrix, num_coefficients, up_sampling_factor, down_sampling_factor):
    """Perform the preprocessing steps of the fast pipeline. Obtain the Mel-frequency cepstral coefficients (MFCCs) from the input signal.
    
    Parameters
    ----------
    - audio_binary [tf.Tensor]: tensor of dtype "string", with the audio file contents.
    - length [int]: frame length.
    - step [int]: frame step.
    - weight_matrix [tf.Tensor of shape [num_spectrogram_bins, num_mel_bins]]: linear to mel weight matrix.
    - num_coefficients [int]: number of coefficients to select.
    - down_sampling_factor [int]: down sampling factor of the audio signal to be applied.

    Output
    ------
    - [tf.Tensor] MFCCs coefficients
    """
    
    audio, frequency = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)

    zero_padding = tf.zeros([frequency] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([frequency])

    # resample the signal
    audio = signal.resample_poly(audio.numpy(), up_sampling_factor, down_sampling_factor)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    frequency = int(frequency*up_sampling_factor/down_sampling_factor)
    num_frames = (frequency - length) // step + 1

    stft = tf.signal.stft(audio, length, step, fft_length=length)
    spectrogram = tf.abs(stft)

    mel_spectrogram = tf.tensordot(spectrogram, weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.reshape(mfccs, [1, num_frames, num_coefficients, 1])
    return mfccs



# definition of the success checker function

def success_checker(prob_distribution, threshold, policy):
    """Success checker function, in order to know whether the prediction based on the fast preprocessing pipleine is confident enough or whether it is better to invoke the web service.
    
    Parameters
    ----------
    - prob_distribution [tf.Tensor]: tensor with the model's output probability distribution over classes.
    - threshold [float]: threshold to decide whether to invoke the web service or not.
    - policy [int, 1 or 2]: which success check policy to use. 1 regards only the top probability value, 2 takes into account the top 2 probability values.
    
    Output
    ------
    - [boolean] a flag about the prediction confidence. If True the prediction based on the fast preprocessing pipline is enugh, if False the web service should be invoked.
    """

    if policy == 1: # look only at the top 1 probability value
        # if the top value is greater than the threshold, the prediction is valid. Else the web service will be invoked.
        if tf.reduce_max(prob_distribution).numpy() > threshold: 
            flag = True
        else:
            flag = False

    elif policy == 2: #  look at the top 2 probability value
        # if the difference is greater than the threshold, the confidence of the prediction is enough. Else the web service will be invoked.
        results = tf.math.top_k(prob_distribution, k = 2).values.numpy()
        if results[0]-results[1] > threshold:
            flag = True
        else:
            flag = False

    else: # if the policy value is not 1 or 2, it is not supported. It can be implemented in this function though.
        print("\nPolicy number not supported. Only 1 and 2 are yet implemented. Set to True by default to optimize inference time and bandwidth.\n")
        flag = True
    
    return flag
    






if __name__ == "__main__":

    # retrieve the mini-speech command data set
    zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')
    
    # retrieve the names of the test files to be processed
    with open('kws_test_split.txt') as f:
        test_files = tf.convert_to_tensor(f.read().splitlines())
    
    # retrieve the labels associated to the classes. If there is a file read the labels from there, else use the default ones.
    if os.path.isfile("labels.txt"): # read from file the labels
        with open("labels.txt", "r") as fl:
            line = fl.read()
            LABELS = line[1:-1].replace("'","").replace(" ","").split(",")
    else: # use the default ones
        LABELS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
    
    # perform grid search
    param_grid = {'num_mel_bins':[20],'UP':[4],'DOWN': [5], "policy":[2], "threshold":[0.1]} 
    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


    # iterate over all the possible parameters combinations
    for config in permutations_dicts:

        print(config)

        # we set the options for the fast preprocessing
        frequency = 16000
        lower_frequency = 20
        upper_frequency = 4000
        num_mel_bins = config['num_mel_bins']
        num_coefficients = 10
        UP = config['UP']
        DOWN = config['DOWN']
        policy = config['policy']
        threshold = config['threshold']
        
        
        url = 'http://localhost:8080/'


        resampled_frequency = int(frequency*UP/DOWN)
        length = int(0.04 * resampled_frequency)
        step = int(0.02 * resampled_frequency)
        num_spectrogram_bins = length // 2 + 1
        
        # compute the matrix to warp linear scale spectrograms to the mel scale
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = num_mel_bins, 
            num_spectrogram_bins = num_spectrogram_bins, 
            sample_rate = resampled_frequency, 
            lower_edge_hertz = lower_frequency,
            upper_edge_hertz = upper_frequency
            )
            
        # load the tlite model
        tflite_model_path = "kws_dscnn_True.tflite"
        interpreter = tflite.Interpreter(model_path = tflite_model_path)
        interpreter.allocate_tensors() # allocate the memory space required for the inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # use a counter to keep track to the number of right predictions on the entire test set, a variable to store the sum of the inference times, the number of calls to the web service and the communication cost
        TP_counter = 0 
        total_inference_time = 0.0
        calls_web_service = 0
        communication_cost = 0.0

    
        # read sequentially the files
        for filename in test_files:
            start_inference = time.time()

            # preprocess one file at a time
            parts = tf.strings.split(filename, os.path.sep)
            true_label = tf.argmax(parts[-2] == LABELS)
            audio_binary = tf.io.read_file(filename)
            input_tensor = preprocess_with_mfcc_fast(audio_binary, length, step, linear_to_mel_weight_matrix, num_coefficients, UP, DOWN)
            # print("Preprocessing :", time.time()-start_inference)

            # run the inference
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke() 
            my_output = interpreter.get_tensor(output_details[0]['index'])[0]
            total_inference_time += time.time() - start_inference

            # apply the softmax on the output scores to transform them into a probability distribution and perform the success check
            sotfmaxed_ouput = tf.nn.softmax(my_output)
            flag = success_checker(sotfmaxed_ouput, threshold, policy)

            # temporary log file
            # top2 = tf.math.top_k(sotfmaxed_ouput, k = 2).values.numpy()
            # tmp_array = np.hstack([sotfmaxed_ouput.numpy(),top2,np.array(top2[0]-top2[1]),np.array(true_label == tf.argmax(sotfmaxed_ouput))])
            # tmp_log_df = pd.DataFrame(data = tmp_array).T
            # tmp_log_df.to_csv("log_file_probabilities.csv",mode='a',header=not(os.path.exists("log_file_probabilities.csv")))
            

            if flag == True:
                # there is enough confidence to provide as ouput the fast pipeline inference
                pred_label = tf.argmax(sotfmaxed_ouput)
            
            else:
                # invoke the web service
                calls_web_service += 1
                
                # retrieve the audio file contents in form of a numpy array
                audio_bytes = audio_binary.numpy()
                
                # conversion to base64 bytes from bytes
                audio_b64bytes = base64.b64encode(audio_bytes)
                
                # conversion to base64 strings
                audio_string = audio_b64bytes.decode()
                timestamp = int(datetime.datetime.now().timestamp()) # posix stored in the timestamp attribute of the object 
                
                # pack into SenML + JSON string to obtain the body of the request
                body = {
                    "bn": "raspberrypi.local", # id
                    "e": [{"n": "audio", "u": "/", "t": timestamp, "vd": audio_string}] #vd because array, v for scalars  
                }
                body = json.dumps(body)

                # update the communication cost value
                communication_cost += sys.getsizeof(body)/1000000
                
                # perform the request 
                r = requests.put(url, data = body)

                if r.status_code == 200:
                    # retrieve the output of the slow preprocessing pipeline based inference from the body of the response
                    body = r.json()
                    pred_label = tf.convert_to_tensor(int(body['label']), dtype = tf.int64)
                else:
                    print("\nERROR. Status code {}: {}".format(r.status_code, r.reason))
                    break

            # if the prediction is correct, update the counter of the number of correct predictions
            if pred_label == true_label:
                TP_counter += 1


        accuracy_avg = TP_counter/len(test_files)*100
        total_time_avg = total_inference_time/len(test_files)*1000
        print("Accuracy: {:.3f}%".format(accuracy_avg))
        print("Total inference time: {:.2f} ms".format(total_time_avg))
        print("Number of calls to the web_service: {}".format(calls_web_service))
        print("Total communication cost: {:.3f} MB".format(communication_cost))

        tmp_df = pd.DataFrame.from_dict([{"num_mel_bins":num_mel_bins,"down":DOWN,"up":UP,"resampled_frequency":resampled_frequency,"accuracy":accuracy_avg,"total_time":total_time_avg, "communication_cost":communication_cost, "number of webservice calls":calls_web_service, "policy":policy, "threshold":threshold}])
        tmp_df.to_csv("log_bea.csv",mode='a',header=not(os.path.exists("log_bea.csv")))