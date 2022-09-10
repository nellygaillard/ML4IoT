import cherrypy
import json
import tensorflow as tf
import tensorflow.lite as tflite
import base64



def preprocess_with_mfcc_slow(audio_bytes, length, step, weight_matrix, num_coefficients):
    """Perform the preprocessing steps of the slow pipeline. Obtain the Mel-frequency cepstral coefficients (MFCCs) from the input signal.
    
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
    
    audio, frequency = tf.audio.decode_wav(audio_bytes)
    audio = tf.squeeze(audio, axis=1)

    zero_padding = tf.zeros([frequency] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([frequency])

    num_frames = (frequency - length) // step + 1

    stft = tf.signal.stft(audio, length, step, fft_length=length)
    spectrogram = tf.abs(stft)

    mel_spectrogram = tf.tensordot(spectrogram, weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.reshape(mfccs, [1, num_frames, num_coefficients, 1])
    return mfccs




class SlowPipelineInference(object):
    exposed = True
    
    def __init__(self):

        # load the tflite model
        self.interpreter = tflite.Interpreter(model_path = "kws_dscnn_True.tflite")
        self.interpreter.allocate_tensors() 
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.frequency = 16000
        self.length =  int(0.04 * self.frequency)
        self.step = int(0.02 * self.frequency)
        
        # compute the weight_matrix only once and store it since it will be used multiple times
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = 40, 
            num_spectrogram_bins = self.length // 2 + 1, 
            sample_rate = self.frequency, 
            lower_edge_hertz = 20,
            upper_edge_hertz = 4000
        )


    def GET(self, *path, **query):
        pass


    def POST(self, *path, **query):
        pass


    def PUT(self, *path, **query):
        # retrieve the audio file from the body of the request
        body = cherrypy.request.body.read()
        body = json.loads(body)
        audio = body['e'][0]['vd']
        audio_bytes = base64.b64decode(audio)

        # perform the preprocessing
        input_tensor = preprocess_with_mfcc_slow(audio_bytes, self.length, self.step, self.linear_to_mel_weight_matrix, 10)
        
        # run the inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke() 
        my_output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # obtain the predicted label and send it back to the device which performed the request
        pred_label = int(tf.argmax(my_output).numpy())
        prob_label = float(my_output[pred_label])
        return json.dumps({"label":pred_label, "probability":prob_label})


    def DELETE(self, *path, **query):
        pass



if __name__ == '__main__':

    # define the dictionary with basic configurations of cherrypy
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    # pass to cherrypy the object that has to be transformed to a webservice
    cherrypy.tree.mount(SlowPipelineInference(), '', conf)
    cherrypy.config.update({'server.socket_host':'0.0.0.0'})
    cherrypy.config.update({'server.socket_port':8080})
    # run the cherrypy engine
    cherrypy.engine.start()
    # after starting cherrypy will be online forever running in background
    cherrypy.engine.block() 