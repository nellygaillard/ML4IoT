import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot 
import zlib 

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds



class KeyWordSpottingModel():
	
    def __init__(self, version):
        
        if version.lower() == 'a':
            self.alpha = 0.8
            self.do_pruning = False
            self.version = version
        elif version.lower() == 'b':
            self.alpha = 0.4
            self.do_pruning = False
            self.version = version
        elif version.lower() == 'c':
            self.alpha = 0.4
            self.do_pruning = True
            self.final_sparsity = 0.65
            self.version = version
        
        self.model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=[61,10,1], filters=int(self.alpha * 256), kernel_size=[3, 3], strides=[2,1], use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                tf.keras.layers.Conv2D(input_shape=[61,10,1], filters=int(self.alpha * 256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, ),
                tf.keras.layers.Conv2D(input_shape=[61,10,1], filters=int(self.alpha * 256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Dense(8)])
        
    def train_base_model(self, train_ds, val_ds, epochs=40):
        """Train the model on the train dataset and validate it on the validation set.
        
        Parameters
        ----------
        - train_ds: train dataset
        - val_ds: validation dataset
        - epochs: number of epochs for the training. If pruning is performed, two trainings are performed. The first is required to initialize the weights, the second to apply pruning."""
        
        self.epochs=epochs

		# compile the model
        self.model.compile(
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer = tf.optimizers.Adam(),
            metrics = [tf.metrics.SparseCategoricalAccuracy()]
            )
            
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor = 'val_sparse_categorical_accuracy', factor = 0.3, patience = 2, min_lr = 1e-04),
            keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',patience = 6, verbose = 0, mode='auto', restore_best_weights=True)
            ]
        
        history = self.model.fit(
            train_ds,
            epochs = epochs,
            validation_data = val_ds,
            callbacks = callbacks
            )
            
        if self.do_pruning: # further training if pruning is applied
        
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude 
            
            self.model = prune_low_magnitude(self.model, pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.30,  
                final_sparsity=self.final_sparsity,
                begin_step=0,
                end_step=np.ceil(len(train_ds)/32).astype(np.int32)*epochs) )
                
            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                keras.callbacks.ReduceLROnPlateau(monitor = 'val_sparse_categorical_accuracy', factor = 0.3, patience = 2, min_lr = 1e-04),
                keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience = 6, verbose = 0, mode='auto', restore_best_weights=True)
            ]
            
            self.model.compile(
                loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer = 'adam',
                metrics = [tf.metrics.SparseCategoricalAccuracy()])
                
            history = self.model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=callbacks)
                
        return history
        
    def evaluate(self, test_ds):
        """Evaluate the model on the test dataset.
        
        Parameters
        ----------
        - test_ds: test dataset on which the model has to be evaluated"""

        return self.model.evaluate(test_ds)
        
        
    def create_tflite(self):
        """Create the TFLite model."""
        
        if self.do_pruning:
            self.model = tfmot.sparsity.keras.strip_pruning(self.model)
            
        self.model.save("model_keyword_spotting_version_{}".format(self.version))
        
        converter = tf.lite.TFLiteConverter.from_saved_model("model_keyword_spotting_version_{}".format(self.version))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert() 
        
        tflite_model_dir = "./Group7_kws_{}.tflite".format(self.version)
        
        with open("{}.zlib".format(tflite_model_dir), 'wb') as f:
            tflite_compressed = zlib.compress(tflite_model)
            f.write(tflite_compressed)

        # with open("{}".format(tflite_model_dir), 'wb') as f:
        #     f.write(tflite_model)
            
        return tflite_model


if __name__ == "__main__":
    
    seed = 31
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True, help='Exercise version: a, b, c')
    args = parser.parse_args()


    zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')


    # read splits from file
    with open('kws_train_split.txt') as f:
        train_files = tf.convert_to_tensor(f.read().splitlines())
    with open('kws_test_split.txt') as f:
        test_files = tf.convert_to_tensor(f.read().splitlines())
    with open('kws_val_split.txt','r') as f:
        val_files = tf.convert_to_tensor(f.read().splitlines())

    LABELS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
    EPOCHS = 40
    options = {'frame_length': 512, 'frame_step': 256, 'mfcc': True, 'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10}

    generator = SignalGenerator(LABELS, 16000, **options)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    test_ds = generator.make_dataset(test_files, False)

    model = KeyWordSpottingModel(args.version)
    history = model.train_base_model(train_ds, val_ds, epochs=EPOCHS)
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy: ", accuracy)

    model_tflite = model.create_tflite()